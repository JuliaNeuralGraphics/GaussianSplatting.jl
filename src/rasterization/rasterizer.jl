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

    antialias::Bool
    fused::Bool
    mode::Symbol
end

function GaussianRasterizer(kab, camera::Camera; kwargs...)
    (; width, height) = resolution(camera)
    GaussianRasterizer(kab; width, height, kwargs...)
end

function GaussianRasterizer(kab;
    width::Int, height::Int,
    antialias::Bool = false,
    fused::Bool = true,
    mode::Symbol = :rgb,
)
    @assert width % 16 == 0 && height % 16 == 0
    antialias && fused && error(
        "`antialias=true` requires `fused=false` for GaussianRasterizer.")

    # TODO support :d
    modes = (:rgb, :rgbd)
    mode in modes || error("Invalid render: $mode ∉ $modes")

    grid = SVector{2, Int32}(cld(width, BLOCK[1]), cld(height, BLOCK[2]))
    istate = ImageState(kab; width, height, grid_size=Int(prod(grid)))
    gstate = GeometryState(kab, 0; extended=mode == :rgbd)
    bstate = BinningState(kab, 0)

    # TODO organize in a render cache/state
    shs = KA.allocate(kab, Float32, (3, 0, 0))
    scales_act = KA.allocate(kab, Float32, (3, 0))
    opacities_act = KA.allocate(kab, Float32, (1, 0))

    image = KA.allocate(kab, Float32, (mode == :rgbd ? 4 : 3, width, height))
    host_image, pinned_image = allocate_pinned(kab, Float32, (3, width, height))

    rast = GaussianRasterizer(
        istate, gstate, bstate,
        shs, scales_act, opacities_act,
        image, pinned_image, host_image,
        grid, antialias, fused, mode)
    finalizer(rast) do rast
        unpin_memory(rast.pinned_image)
    end
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

include("projection.jl")
include("spherical_harmonics.jl")
include("render.jl")

# OpenGL convertions.

function gl_texture(r::GaussianRasterizer)
    r.pinned_image .= @view(r.image[1:3, :, :])
    KA.synchronize(get_backend(r))
    clamp01!(r.host_image)
    reverse!(r.host_image; dims=3)
    return r.host_image
end

function gl_depth(r::GaussianRasterizer)
    # Copying just 1 single element is 5x slower, so we copy 3.
    r.pinned_image .= @view(r.image[2:4, :, :])
    KA.synchronize(get_backend(r))

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

to_image(x) = colorview(RGB, permutedims(clamp01!(x), (1, 3, 2)))

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
)
    # If rendering outside AD, use non-allocating path.
    inplace = NNlib.within_gradient(means_3d)

    shs = if inplace
        isempty(sh_remainder) ? sh_color : hcat(sh_color, sh_remainder)
    else
        n_features = size(sh_color, 2) + size(sh_remainder, 2)
        target_size = (3, n_features, size(sh_color, 3))
        rshs = cache!(rast, :shs, target_size)
        rshs[:, 1:1, :] .= sh_color
        isempty(sh_remainder) || (rshs[:, 2:end, :] .= sh_remainder)
        rshs
    end

    opacities_act = if inplace
        NU.sigmoid.(opacities)
    else
        ropacities = cache!(rast, :opacities_act, size(opacities))
        ropacities .= NU.sigmoid.(opacities)
    end

    scales_act = if inplace
        exp.(scales)
    else
        rscales = cache!(rast, :scales_act, size(scales))
        rscales .= exp.(scales)
    end

    if rast.fused
        return rasterize(
            means_3d, shs, opacities_act, scales_act, rotations, R_w2c, t_w2c;
            rast, camera, sh_degree, background)
    else
        means_2d, conics, compensations, depths = project(
            means_3d, scales_act, rotations;
            rast, camera, near_plane=0.2f0, far_plane=1000f0,
            radius_clip=3f0, blur_ϵ=0.3f0)

        colors = spherical_harmonics(means_3d, shs; rast, camera, sh_degree)

        # TODO handle :d mode
        color_features = if rast.mode == :rgbd
            vcat(colors, reshape(depths, 1, :))
        else
            colors
        end

        opacities_scaled = if rast.antialias
            opacities_act .* compensations
        else
            opacities_act
        end

        image = render(
            means_2d, conics, opacities_scaled, color_features;
            rast, camera, background, depths)
        return image
    end
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
)
    render_depth = rast.mode == :rgbd
    render_depth && @assert rast.gstate.color_features ≢ nothing

    kab = get_backend(rast)
    n = size(means_3d, 2)
    if length(rast.gstate) < n
        # @info "[rast.gstate] resize: $(length(rast.gstate)) -> $n"
        KA.unsafe_free!(rast.gstate)

        do_record = record_memory(kab)
        do_record && record_memory!(kab, false; free=false)
        rast.gstate = GeometryState(kab, n; extended=render_depth)
        do_record && record_memory!(kab, true)
    end

    (; width, height) = resolution(camera)
    @assert width % 16 == 0 && height % 16 == 0

    fill!(rast.image, 0f0)

    K = camera.intrinsics
    R = R_w2c ≡ nothing ?
        SMatrix{3, 3, Float32}(camera.w2c[1:3, 1:3]) :
        R_w2c
    t = t_w2c ≡ nothing ?
        SVector{3, Float32}(camera.w2c[1:3, 4]) :
        t_w2c

    # TODO make configurable.
    near_plane, far_plane = 0.2f0, 1000f0
    radius_clip = 3f0 # In pixels.
    blur_ϵ = 0.3f0

    project!(kab)(
        # Output.
        rast.gstate.depths,
        rast.gstate.radii,
        rast.gstate.means_2d,
        rast.gstate.conic_opacities,
        nothing, # compensations
        # Input Gaussians.
        _as_T(SVector{3, Float32}, means_3d),
        _as_T(SVector{3, Float32}, scales),
        _as_T(SVector{4, Float32}, rotations),
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
        rast.gstate.radii,
        rast.grid, BLOCK; ndrange=n)

    cumsum!(
        @view(rast.gstate.points_offset[1:n]),
        @view(rast.gstate.tiles_touched[1:n]))
    # Get total number of tiles touched.
    n_rendered = Int(@allowscalar rast.gstate.points_offset[n])
    n_rendered == 0 && return rast.image

    if length(rast.bstate) < n_rendered
        # @info "[rast.bstate] resize: $(length(rast.bstate)) -> $n_rendered"
        KA.unsafe_free!(rast.bstate)

        do_record = record_memory(kab)
        do_record && record_memory!(kab, false; free=false)
        rast.bstate = BinningState(kab, n_rendered)
        do_record && record_memory!(kab, true)
    end

    # For each instance to be rendered, produce [tile | depth] key
    # and corresponding duplicated Gaussian indices to be sorted.
    duplicate_with_keys!(kab)(
        # Output.
        rast.bstate.gaussian_keys_unsorted,
        rast.bstate.gaussian_values_unsorted,
        # Input.
        rast.gstate.means_2d,
        rast.gstate.depths,
        rast.gstate.points_offset,
        rast.gstate.radii, rast.grid, BLOCK; ndrange=n)

    if use_ak(kab)
        sortperm!(
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

    color_features = if render_depth
        rast.gstate.color_features[1:3, :] .= reshape(reinterpret(Float32, rast.gstate.rgbs), 3, :)
        rast.gstate.color_features[4, :] .= rast.gstate.depths
        _as_T(SVector{4, Float32}, rast.gstate.color_features)
    else
        rast.gstate.rgbs
    end

    render!(kab, (Int.(BLOCK)...,), (width, height))(
        # Outputs.
        rast.image, rast.istate.n_contrib, rast.istate.accum_α,
        # Inputs.
        rast.bstate.gaussian_values_sorted,
        rast.gstate.means_2d,
        opacities,
        rast.gstate.conic_opacities,
        color_features,
        rast.istate.ranges,
        SVector{2, Int32}(width, height),
        render_depth ? SVector{4, Float32}(background..., 0f0) : background,
        BLOCK, Val(BLOCK_SIZE))
    return rast.image
end

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
    render_depth = rast.mode == :rgbd
    kab = get_backend(rast)
    n = size(means_3d, 2)

    (; width, height) = resolution(camera)
    @assert width % 16 == 0 && height % 16 == 0

    vcolor_features = KA.zeros(kab, Float32, (render_depth ? 4 : 3, n))
    vconics = KA.zeros(kab, Float32, (3, n))
    vcov = KA.zeros(kab, Float32, (6, n))

    vmeans = KA.zeros(kab, Float32, (3, n))
    vshs = KA.zeros(kab, Float32, size(shs))
    vopacities = KA.zeros(kab, Float32, (1, n))
    vscales = KA.zeros(kab, Float32, (3, n))
    vrot = KA.zeros(kab, Float32, (4, n))
    fill!(reinterpret(Float32, rast.gstate.∇means_2d), 0f0)

    K = camera.intrinsics
    (; width, height) = resolution(camera)

    color_features = render_depth ?
        _as_T(SVector{4, Float32}, rast.gstate.color_features) :
        rast.gstate.rgbs

    ∇render!(kab, (Int.(BLOCK)...,), (width, height))(
        # Outputs.
        vcolor_features,
        vopacities,
        vconics,
        reshape(reinterpret(Float32, rast.gstate.∇means_2d), 2, :),
        # Inputs.
        reshape(
            reinterpret(SVector{render_depth ? 4 : 3, Float32}, vpixels),
            size(vpixels)[2:3]),
        rast.istate.n_contrib,
        rast.istate.accum_α,

        rast.bstate.gaussian_values_sorted,
        rast.gstate.means_2d,
        opacities,
        rast.gstate.conic_opacities,
        color_features,

        rast.istate.ranges,
        SVector{2, Int32}(width, height),
        render_depth ? SVector{4, Float32}(background..., 0f0) : background,
        rast.grid, BLOCK, Val(BLOCK_SIZE))

    vrgbs, vdepths = if render_depth
        vcolor_features[1:3, :], vcolor_features[4, :]
    else
        vcolor_features, nothing
    end

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

    return vmeans, vshs, vopacities, vscales, vrot, vR, vt
end

function ChainRulesCore.rrule(::typeof(rasterize),
    means_3d, shs, opacities, scales, rotations,
    R_w2c = nothing, t_w2c = nothing;
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
    background::SVector{3, Float32},
)
    image = rasterize(
        means_3d, shs, opacities, scales, rotations, R_w2c, t_w2c;
        rast, camera, sh_degree, background)

    function _pullback(vpixels)
        ∇ = ∇rasterize(
            vpixels, means_3d, shs, scales, rotations, opacities,
            rast.gstate.radii, R_w2c, t_w2c; rast, camera, sh_degree, background)
        return (NoTangent(), ∇...)
    end
    return image, _pullback
end
