# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
include("states.jl")
include("utils.jl")

mutable struct GaussianRasterizer{
    I <: ImageState,
    G <: GeometryState,
    B <: BinningState,
    K <: AbstractArray{Float32, 3},
    P <: AbstractArray{Float32, 3},
    S <: AbstractMatrix{Float32},
}
    istate::I
    gstate::G
    bstate::B

    # Temporary storage to avoid allocating during rendering outside of AD.
    shs::K
    scales_act::S
    opacities_act::S

    image::K
    pinned_image::P
    host_image::Array{Float32, 3}

    grid::SVector{2, Int32}

    # Camera-space depth range Gaussians are kept in (see `project!`).
    # The sky dome needs a far plane past its radius, so this is per-rasterizer
    # rather than a global constant.
    near_plane::Float32
    far_plane::Float32

    mode::Symbol

    # Whether the blend backward also accumulates `gstate.∇means_2d_abs`.
    # Costs two extra atomics per gaussian-pixel, so it is opt-in: only
    # `ImprovedGSStrategy` consumes it (see `enable_abs_grad!`).
    abs_grad::Bool

    # Which blend backward `∇rasterize` launches:
    #   `:per_pixel` — `∇render!`, one thread per pixel, atomics per fragment.
    #   `:per_splat` — `∇render_wavefront!`, one lane per splat, atomics per (tile, splat).
    backward::Symbol
end

"""
Make `∇rasterize` accumulate `gstate.∇means_2d_abs` (`Σₚ |∂L/∂μ|`) on every
backward pass. Off by default; a `Trainer` turns it on for the strategies that
densify from the absolute image-space gradient.
"""
enable_abs_grad!(rast::GaussianRasterizer) = (rast.abs_grad = true; rast)

function GaussianRasterizer(kab, camera::Camera; kwargs...)
    (; width, height) = resolution(camera)
    GaussianRasterizer(kab; width, height, kwargs...)
end

"""
Number of blended feature channels rendered by `mode`:
`:rgb` → 3, `:rgbd` → + depth & alpha, `:rgbdn` → + camera-space normal.
See `GeometryState` for the channel layout.
"""
n_color_features(mode::Symbol) =
    mode == :rgb ? 3 :
    mode == :rgbd ? 5 :
    mode == :rgbdn ? 8 :
    error("Invalid render mode: `$mode`.")

"""
Rasterizer mode a `Trainer` needs for the given optimization params:
the depth-normal consistency loss requires a rendered normal channel.
"""
training_rasterizer_mode(opt_params::OptimizationParams) =
    opt_params.use_normal_loss ? :rgbdn : :rgbd

"""
The ndrange that covers `width × height` in whole tiles.
Round frame tiles *up*, `render!` & `∇render!` handle out-of-bounds access
and those threads just help with prefetching the data.
"""
tile_ndrange(width::Int, height::Int) = (
    cld(width, BLOCK[1]) * BLOCK[1],
    cld(height, BLOCK[2]) * BLOCK[2])

function GaussianRasterizer(kab;
    width::Int, height::Int,
    mode::Symbol = :rgbd,
    near_plane::Float32 = 0.2f0,
    far_plane::Float32 = 1000f0,
    backward::Symbol = :per_splat,
)
    modes = (:rgb, :rgbd, :rgbdn)
    mode in modes || error("Invalid render: $mode ∉ $modes")
    backwards = (:per_pixel, :per_splat)
    backward in backwards || error("Invalid backward: $backward ∉ $backwards")

    grid = SVector{2, Int32}(cld(width, BLOCK[1]), cld(height, BLOCK[2]))
    istate = ImageState(kab; width, height, grid_size=Int(prod(grid)))
    gstate = GeometryState(kab, 0; n_features=n_color_features(mode))
    bstate = BinningState(kab, 0)

    # TODO organize in a render cache/state
    shs = KA.allocate(kab, Float32, (3, 0, 0))
    scales_act = KA.allocate(kab, Float32, (3, 0))
    opacities_act = KA.allocate(kab, Float32, (1, 0))

    image = KA.allocate(kab, Float32, (n_color_features(mode), width, height))
    host_image, pinned_image = allocate_pinned(kab, Float32, (3, width, height))

    rast = GaussianRasterizer(
        istate, gstate, bstate,
        shs, scales_act, opacities_act,
        image, pinned_image, host_image,
        grid, near_plane, far_plane, mode, false, backward)
    finalizer(rast -> unpin_memory(rast.pinned_image), rast)
    return rast
end

function cache!(rast::GaussianRasterizer, name::Symbol, target_size)
    x = getproperty(rast, name)
    if size(x) != target_size
        KA.unsafe_free!(x)
        x = KA.allocate(get_backend(rast), eltype(x), target_size)
        setproperty!(rast, name, x)
    end
    return x
end

KernelAbstractions.get_backend(r::GaussianRasterizer) = get_backend(r.image)

"""
Release the scene-sized scratch buffers: they grow with the number of
Gaussians & rendered tile instances and are never shrunk, so they are what
a closed scene leaves behind.

The rasterizer stays usable — the next render reallocates them for the new scene.
"""
function release_scene_buffers!(rast::GaussianRasterizer)
    kab = get_backend(rast)

    KA.unsafe_free!(rast.gstate)
    KA.unsafe_free!(rast.bstate)
    rast.gstate = GeometryState(kab, 0; n_features=n_color_features(rast.mode))
    rast.bstate = BinningState(kab, 0)

    cache!(rast, :shs, (3, 0, 0))
    cache!(rast, :scales_act, (3, 0))
    cache!(rast, :opacities_act, (1, 0))
    return
end

# `pinned_image` is excluded: it is registered host memory, not a device
# allocation.
memory_usage(rast::GaussianRasterizer) =
    memory_usage(rast.istate) +
    memory_usage(rast.gstate) +
    memory_usage(rast.bstate) +
    memory_usage(rast.shs) +
    memory_usage(rast.scales_act) +
    memory_usage(rast.opacities_act) +
    memory_usage(rast.image)

function KA.unsafe_free!(rast::GaussianRasterizer)
    KA.unsafe_free!(rast.istate)
    KA.unsafe_free!(rast.gstate)
    KA.unsafe_free!(rast.bstate)
    KA.unsafe_free!(rast.shs)
    KA.unsafe_free!(rast.scales_act)
    KA.unsafe_free!(rast.opacities_act)
    KA.unsafe_free!(rast.image)
    return
end

include("projection.jl")
include("spherical_harmonics.jl")
include("render.jl")

# OpenGL convertions.
# Both return the rasterizer-owned `host_image`, which the next render
# overwrites: callers must copy it out if they keep it around.

function gl_texture(r::GaussianRasterizer)
    r.pinned_image .= @view(r.image[1:3, :, :])
    blocking_synchronize(get_backend(r))
    clamp01!(r.host_image)
    reverse!(r.host_image; dims=3)
    return r.host_image
end

function gl_depth(r::GaussianRasterizer)
    # Copying just 1 single element is 5x slower, so we copy 3.
    r.pinned_image .= @view(r.image[2:4, :, :])
    blocking_synchronize(get_backend(r))

    r.host_image[1:2, :, :] .= @view(r.host_image[[3], :, :])

    # Normalization value was chosen based on the visual output.
    r.host_image .*= 1f0 / 50f0
    clamp01!(r.host_image)
    reverse!(r.host_image; dims=3)
    return r.host_image
end

# Image conversions.

function to_image(r::GaussianRasterizer)
    host_image = clamp01!(@view(Array(r.image)[1:3, :, :]))
    return colorview(RGB, permutedims(host_image, (1, 3, 2)))
end

function to_depth(r::GaussianRasterizer)
    depth_image = transpose(Array(r.image)[4, :, :])
    depth_image .*= 1f0 / 50f0
    clamp01!(depth_image)
    return colorview(Gray, depth_image)
end

to_image(x) = colorview(RGB, permutedims(clamp01!(x), (1, 3, 2)))

function to_depth(x)
    depth_image = transpose(x)
    depth_image .*= 1f0 / 50f0
    clamp01!(depth_image)
    return colorview(Gray, depth_image)
end

function (rast::GaussianRasterizer)(
    means_3d::AbstractMatrix{Float32},
    opacities::AbstractMatrix{Float32},
    scales::AbstractMatrix{Float32},
    rotations::AbstractMatrix{Float32},
    sh_color::AbstractArray{Float32, 3},
    sh_remainder::AbstractArray{Float32, 3},
    R_w2c = nothing, t_w2c = nothing;
    camera::Camera, sh_degree::Int,
    background::SVector{3, Float32} = zeros(SVector{3, Float32}),

    covisibilities::Maybe{AbstractVector{Bool}} = nothing,
    uncertainties::Maybe{AbstractMatrix{Float32}} = nothing,
    edge_scores::Maybe{AbstractVector{Float32}} = nothing,
    edge_map::Maybe{AbstractMatrix{Float32}} = nothing,
)
    # If rendering outside AD, use non-allocating path.
    within_AD = within_gradient(means_3d)

    shs = if within_AD
        isempty(sh_remainder) ? sh_color : hcat(sh_color, sh_remainder)
    else
        n_features = size(sh_color, 2) + size(sh_remainder, 2)
        target_size = (3, n_features, size(sh_color, 3))
        rshs = cache!(rast, :shs, target_size)
        rshs[:, 1:1, :] .= sh_color
        isempty(sh_remainder) || (rshs[:, 2:end, :] .= sh_remainder)
        rshs
    end

    opacities_act = if within_AD
        NU.sigmoid.(opacities)
    else
        ropacities = cache!(rast, :opacities_act, size(opacities))
        ropacities .= NU.sigmoid.(opacities)
    end

    isotropic = size(scales, 1) == 1
    scales_act = if within_AD
        exp.(isotropic ? vcat(scales, scales, scales) : scales)
    else
        rscales = cache!(rast, :scales_act, (3, size(scales, 2)))
        if isotropic
            rscales[[1], :] .= exp.(scales)
            rscales[2, :] .= rscales[1, :]
            rscales[3, :] .= rscales[1, :]
            rscales
        else
            rscales .= exp.(scales)
        end
    end

    return rasterize(
        means_3d, shs, opacities_act, scales_act, rotations, R_w2c, t_w2c;
        rast, camera, sh_degree, background, covisibilities, uncertainties,
        edge_scores, edge_map)
end

function rasterize(
    means_3d::AbstractMatrix{Float32},
    shs::AbstractArray{Float32, 3},
    opacities::AbstractMatrix{Float32},
    scales::AbstractMatrix{Float32},
    rotations::AbstractMatrix{Float32},
    R_w2c = nothing, t_w2c = nothing;
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
    background::SVector{3, Float32},

    covisibilities::Maybe{AbstractVector{Bool}} = nothing,
    uncertainties::Maybe{AbstractMatrix{Float32}} = nothing,
    edge_scores::Maybe{AbstractVector{Float32}} = nothing,
    edge_map::Maybe{AbstractMatrix{Float32}} = nothing,
)
    channels = n_color_features(rast.mode)
    render_depth = channels > 3
    render_normals = channels > 5
    render_depth && @assert rast.gstate.color_features ≢ nothing

    kab = get_backend(rast)
    n = size(means_3d, 2)
    if length(rast.gstate) < n
        KA.unsafe_free!(rast.gstate)
        rast.gstate = GPUArrays.@uncached GeometryState(kab, n; n_features=channels)
    end

    (; width, height) = resolution(camera)
    fill!(rast.image, 0f0)

    K = camera.intrinsics
    R = R_w2c ≡ nothing ?
        SMatrix{3, 3, Float32}(camera.w2c[1:3, 1:3]) :
        R_w2c
    t = t_w2c ≡ nothing ?
        SVector{3, Float32}(camera.w2c[1:3, 4]) :
        t_w2c

    near_plane, far_plane = rast.near_plane, rast.far_plane
    radius_clip = Int32(3) # In pixels.
    blur_ϵ = 0.3f0

    project!(kab)(
        # Output.
        rast.gstate.depths,
        rast.gstate.radii,
        rast.gstate.means_2d,
        rast.gstate.conic_opacities,
        nothing, # compensations
        render_normals ? rast.gstate.normals : nothing,
        # Input Gaussians.
        _as_T(SVector{3, Float32}, means_3d),
        _as_T(SVector{3, Float32}, scales),
        _as_T(SVector{4, Float32}, rotations),
        opacities,
        # Input camera properties.
        R, t, K.focal, Int32.(K.resolution), K.principal,
        # Config.
        near_plane, far_plane, radius_clip, blur_ϵ; ndrange=n)

    spherical_harmonics!(kab)(
        # Output.
        rast.gstate.rgbs,
        rast.gstate.clamped,
        # Input.
        rast.gstate.radii,
        _as_T(SVector{3, Float32}, means_3d),
        camera.camera_center,
        reinterpret(SVector{3, Float32}, reshape(shs, :, n)),
        Val(sh_degree); ndrange=n)

    count_tiles_per_gaussian!(kab)(
        # Output.
        rast.gstate.tiles_touched,
        # Input.
        rast.gstate.means_2d,
        rast.gstate.conic_opacities,
        opacities,
        rast.gstate.radii,
        rast.grid, BLOCK; ndrange=n)

    cumsum!(
        @view(rast.gstate.points_offset[1:n]),
        @view(rast.gstate.tiles_touched[1:n]))
    # Get total number of tiles touched.
    n_rendered = Int(@allowscalar rast.gstate.points_offset[n])
    n_rendered == 0 && return rast.image

    if length(rast.bstate) < n_rendered
        KA.unsafe_free!(rast.bstate)
        rast.bstate = GPUArrays.@uncached BinningState(kab, n_rendered)
    end

    # For each instance to be rendered, produce [tile | depth] key
    # and corresponding duplicated Gaussian indices to be sorted.
    duplicate_with_keys!(kab)(
        # Output.
        rast.bstate.gaussian_keys_unsorted,
        rast.bstate.gaussian_values_unsorted,
        # Input.
        rast.gstate.means_2d,
        rast.gstate.conic_opacities,
        opacities,
        rast.gstate.depths,
        rast.gstate.points_offset,
        rast.gstate.radii, rast.grid, BLOCK; ndrange=n)

    if use_ak(kab)
        AK.sortperm!(
            @view(rast.bstate.permutation[1:n_rendered]),
            @view(rast.bstate.gaussian_keys_unsorted[1:n_rendered]);
            temp=@view(rast.bstate.permutation_tmp[1:n_rendered]))
    else
        sortperm!(
            @view(rast.bstate.permutation[1:n_rendered]),
            @view(rast.bstate.gaussian_keys_unsorted[1:n_rendered]))
    end
    _permute!(kab)(
        rast.bstate.gaussian_keys_sorted, rast.bstate.gaussian_keys_unsorted,
        rast.bstate.permutation; ndrange=n_rendered)
    _permute!(kab)(
        rast.bstate.gaussian_values_sorted, rast.bstate.gaussian_values_unsorted,
        rast.bstate.permutation; ndrange=n_rendered)

    # Identify start-end of per-tile workloads in sorted keys.
    fill!(rast.istate.ranges, 0u32)
    identify_tile_range!(kab)(
        rast.istate.ranges, rast.bstate.gaussian_keys_sorted;
        ndrange=n_rendered)

    if render_depth
        rast.gstate.color_features[1:3, :] .= reshape(reinterpret(Float32, rast.gstate.rgbs), 3, :)
        rast.gstate.color_features[4, :] .= rast.gstate.depths
        rast.gstate.color_features[5, :] .= 1f0 # Constant feature: renders the alpha map.
        render_normals && (rast.gstate.color_features[6:8, :] .=
            reshape(reinterpret(Float32, rast.gstate.normals), 3, :))
    end
    # Literal channel counts: `render!` is specialized on them.
    color_features =
        render_normals ? _as_T(SVector{8, Float32}, rast.gstate.color_features) :
        render_depth ? _as_T(SVector{5, Float32}, rast.gstate.color_features) :
        rast.gstate.rgbs

    render!(kab, (Int.(BLOCK)...,), tile_ndrange(width, height))(
        # Outputs.
        rast.image, rast.istate.n_contrib, rast.istate.accum_α,
        covisibilities, uncertainties, edge_scores,
        # Inputs.
        edge_map,
        rast.bstate.gaussian_values_sorted,
        rast.gstate.means_2d,
        opacities,
        rast.gstate.conic_opacities,
        color_features,
        rast.istate.ranges,
        SVector{2, Int32}(width, height),
        feature_background(background, channels),
        BLOCK, Val(BLOCK_SIZE))
    return rast.image
end

# Zero background for the depth, alpha & normal channels.
feature_background(background::SVector{3, Float32}, channels::Int) =
    channels == 8 ? SVector{8, Float32}(background..., 0f0, 0f0, 0f0, 0f0, 0f0) :
    channels == 5 ? SVector{5, Float32}(background..., 0f0, 0f0) :
    background

function ∇rasterize(
    vpixels::AbstractArray{Float32, 3},
    means_3d::AbstractMatrix{Float32},
    shs::AbstractArray{Float32, 3},
    scales::AbstractMatrix{Float32},
    rotations::AbstractMatrix{Float32},
    opacities::AbstractMatrix{Float32},
    radii::AbstractVector{Int32},
    R_w2c = nothing, t_w2c = nothing;
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
    background::SVector{3, Float32},
)
    channels = n_color_features(rast.mode)
    render_depth = channels > 3
    render_normals = channels > 5
    kab = get_backend(rast)
    n = size(means_3d, 2)

    (; width, height) = resolution(camera)
    vcolor_features = KA.zeros(kab, Float32, (channels, n))
    vconics = KA.zeros(kab, Float32, (3, n))
    vcov = KA.zeros(kab, Float32, (6, n))

    vmeans = KA.zeros(kab, Float32, (3, n))
    vshs = KA.zeros(kab, Float32, size(shs))
    vopacities = KA.zeros(kab, Float32, (1, n))
    vscales = KA.zeros(kab, Float32, (3, n))
    vrot = KA.zeros(kab, Float32, (4, n))
    fill!(reinterpret(Float32, rast.gstate.∇means_2d), 0f0)
    vmeans_2d_abs = if rast.abs_grad
        buf = reshape(reinterpret(Float32, rast.gstate.∇means_2d_abs), 2, :)
        fill!(buf, 0f0)
        buf
    else
        nothing
    end

    K = camera.intrinsics
    (; width, height) = resolution(camera)

    color_features =
        render_normals ? _as_T(SVector{8, Float32}, rast.gstate.color_features) :
        render_depth ? _as_T(SVector{5, Float32}, rast.gstate.color_features) :
        rast.gstate.rgbs
    vpixel_features =
        render_normals ? reshape(reinterpret(SVector{8, Float32}, vpixels), size(vpixels)[2:3]) :
        render_depth ? reshape(reinterpret(SVector{5, Float32}, vpixels), size(vpixels)[2:3]) :
        reshape(reinterpret(SVector{3, Float32}, vpixels), size(vpixels)[2:3])

    # Both kernels take the same arguments; only the launch geometry differs.
    ∇args = (
        # Outputs.
        vcolor_features,
        vopacities,
        vconics,
        reshape(reinterpret(Float32, rast.gstate.∇means_2d), 2, :),
        vmeans_2d_abs,
        # Inputs.
        vpixel_features,
        rast.istate.n_contrib,
        rast.istate.accum_α,

        rast.bstate.gaussian_values_sorted,
        rast.gstate.means_2d,
        opacities,
        rast.gstate.conic_opacities,
        color_features,

        rast.istate.ranges,
        SVector{2, Int32}(width, height),
        feature_background(background, channels),
        rast.grid, BLOCK)

    if rast.backward == :per_splat
        gsz = backward_group_size(kab)
        # `ndrange` as a keyword, not the 3-arg form used elsewhere in this
        # file: that one makes `ndrange` a type parameter & would recompile the
        # kernel for every distinct tile count.
        ∇render_wavefront!(kab, gsz)(
            ∇args..., Val(BLOCK_SIZE), Val(gsz);
            ndrange=gsz * Int(prod(rast.grid)))
    else
        ∇render!(kab, (Int.(BLOCK)...,), tile_ndrange(width, height))(
            ∇args..., Val(BLOCK_SIZE))
    end

    # Row 5 of `vcolor_features` is the cotangent of the constant-1 alpha
    # feature: dropped (the feature is not a parameter). The alpha map's
    # effect on opacities/means/conics is already accumulated by `∇render!`
    # through the per-channel `vα` path.
    vrgbs, vdepths = if render_depth
        vcolor_features[1:3, :], vcolor_features[4, :]
    else
        vcolor_features, nothing
    end
    vnormals_buf = render_normals ? vcolor_features[6:8, :] : nothing
    vnormals = isnothing(vnormals_buf) ?
        nothing : _as_T(SVector{3, Float32}, vnormals_buf)

    R, t, vR, vt = if R_w2c ≡ nothing
        tmp_R = SMatrix{3, 3, Float32}(camera.w2c[1:3, 1:3])
        tmp_t = SVector{3, Float32}(camera.w2c[1:3, 4])
        tmp_R, tmp_t, nothing, nothing
    else
        tmp_vR = KA.zeros(kab, Float32, 3, 3)
        tmp_vt = KA.zeros(kab, Float32, 3)
        R_w2c, t_w2c, tmp_vR, tmp_vt
    end

    blur_ϵ = 0.3f0
    ∇project!(kab)(
        # Output.
        _as_T(SVector{3, Float32}, vmeans),
        _as_T(SVector{3, Float32}, vscales),
        _as_T(SVector{4, Float32}, vrot),
        vR, vt,

        # Input grad outputs.
        rast.gstate.∇means_2d,
        _as_T(SVector{3, Float32}, vconics),
        nothing, # vcompensations
        vdepths,
        vnormals,

        rast.gstate.conic_opacities,
        rast.gstate.radii,
        # Input Gaussians.
        _as_T(SVector{3, Float32}, means_3d),
        _as_T(SVector{3, Float32}, scales),
        _as_T(SVector{4, Float32}, rotations),
        nothing, # compensations
        # Input camera properties.
        R, t, K.focal, Int32.(K.resolution), K.principal, blur_ϵ; ndrange=n)

    ∇spherical_harmonics!(kab)(
        # Output.
        reinterpret(SVector{3, Float32}, reshape(vshs, :, n)),
        _as_T(SVector{3, Float32}, vmeans),
        # Input.
        _as_T(SVector{3, Float32}, means_3d),
        reinterpret(SVector{3, Float32}, reshape(shs, :, n)),
        rast.gstate.clamped,
        _as_T(SVector{3, Float32}, vrgbs),
        camera.camera_center,
        Val(sh_degree); ndrange=n)

    KA.unsafe_free!(vcolor_features)
    KA.unsafe_free!(vconics)
    KA.unsafe_free!(vcov)
    KA.unsafe_free!(vrgbs)
    isnothing(vdepths) || KA.unsafe_free!(vdepths)
    isnothing(vnormals_buf) || KA.unsafe_free!(vnormals_buf)

    return vmeans, vshs, vopacities, vscales, vrot, vR, vt
end

function ChainRulesCore.rrule(::typeof(rasterize),
    means_3d, shs, opacities, scales, rotations,
    R_w2c = nothing, t_w2c = nothing;
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
    background::SVector{3, Float32},

    covisibilities::Maybe{AbstractVector{Bool}} = nothing,
    uncertainties::Maybe{AbstractMatrix{Float32}} = nothing,
    edge_scores::Maybe{AbstractVector{Float32}} = nothing,
    edge_map::Maybe{AbstractMatrix{Float32}} = nothing,
)
    image = rasterize(
        means_3d, shs, opacities, scales, rotations, R_w2c, t_w2c;
        rast, camera, sh_degree, background, covisibilities, uncertainties,
        edge_scores, edge_map)

    function _pullback(vpixels)
        ∇ = ∇rasterize(
            unthunk(vpixels),
            means_3d, shs, scales, rotations, opacities,
            rast.gstate.radii, R_w2c, t_w2c; rast, camera, sh_degree, background)
        return (NoTangent(), ∇...)
    end
    return image, _pullback
end
