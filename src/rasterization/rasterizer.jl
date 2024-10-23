# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
include("states.jl")
include("projection.jl")
include("spherical_harmonics.jl")
include("utils.jl")
include("forward.jl")
include("backward.jl")

mutable struct GaussianRasterizer{
    I <: ImageState,
    G <: GeometryState,
    B <: BinningState,
    K <: AbstractArray{Float32, 3},
}
    istate::I
    gstate::G
    bstate::B

    image::K
    host_image::Array{Float32, 3}

    grid::SVector{2, Int32}
end

function GaussianRasterizer(kab, camera::Camera)
    (; width, height) = resolution(camera)
    GaussianRasterizer(kab; width, height)
end

function GaussianRasterizer(kab; width::Int, height::Int)
    @assert width % 16 == 0 && height % 16 == 0

    grid = SVector{2, Int32}(cld(width, BLOCK[1]), cld(height, BLOCK[2]))
    istate = ImageState(kab; width, height, grid_size=Int(prod(grid)))
    gstate = GeometryState(kab, 0)
    bstate = BinningState(kab, 0)

    image = KA.allocate(kab, Float32, (3, width, height))
    host_image = Array{Float32, 3}(undef, (3, width, height))
    GaussianRasterizer(istate, gstate, bstate, image, host_image, grid)
end

KernelAbstractions.get_backend(r::GaussianRasterizer) = get_backend(r.image)

# OpenGL convertions.

function gl_texture(r::GaussianRasterizer)
    copyto!(r.host_image, @view(r.image[:, :, end:-1:1]))
    clamp01!(r.host_image)
    return r.host_image
end

# Image conversions.

function to_image(r::GaussianRasterizer)
    raw_img = clamp01!(permutedims(Array(r.image), (1, 3, 2)))
    return colorview(RGB, raw_img)
end

function (rast::GaussianRasterizer)(
    means_3d, opacities, scales, rotations, shs;
    camera::Camera, sh_degree::Int,
    background::SVector{3, Float32} = zeros(SVector{3, Float32}),
)
    rasterize(
        means_3d, shs,
        NU.sigmoid.(opacities),
        exp.(scales),
        rotations;
        rast, camera, sh_degree, background)
end

function rasterize(
    means_3d::AbstractMatrix{Float32},
    shs::AbstractArray{Float32, 3},
    opacities::AbstractMatrix{Float32},
    scales::AbstractMatrix{Float32},
    rotations::AbstractMatrix{Float32};
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
    background::SVector{3, Float32},
)
    kab = get_backend(rast)
    n = size(means_3d, 2)
    length(rast.gstate) == n || (rast.gstate = GeometryState(kab, n))

    (; width, height) = resolution(camera)
    @assert width % 16 == 0 && height % 16 == 0

    _as_T(T, x) = reinterpret(T, reshape(x, :))
    fill!(rast.image, 0f0)

    K = camera.intrinsics
    R_w2c = SMatrix{3, 3, Float32}(camera.w2c[1:3, 1:3])
    t_w2c = SVector{3, Float32}(camera.w2c[1:3, 4])

    # TODO make configurable.
    near_plane, far_plane = 0.2f0, 1000f0
    radius_clip = 3f0 # In pixels.
    blur_ϵ = 0.3f0

    project!(kab, Int(BLOCK_SIZE))(
        # Output.
        rast.gstate.depths,
        rast.gstate.radii,
        rast.gstate.means_2d,
        rast.gstate.conic_opacities,
        # Input Gaussians.
        _as_T(SVector{3, Float32}, means_3d),
        _as_T(SVector{3, Float32}, scales),
        _as_T(SVector{4, Float32}, rotations),
        # Input camera properties.
        R_w2c, t_w2c, K.focal, Int32.(K.resolution), K.principal,
        # Config.
        near_plane, far_plane, radius_clip, blur_ϵ; ndrange=n)

    spherical_harmonics!(kab, Int(BLOCK_SIZE))(
        # Output.
        rast.gstate.rgbs,
        rast.gstate.clamped,
        # Input.
        rast.gstate.radii,
        _as_T(SVector{3, Float32}, means_3d),
        camera.camera_center,
        reinterpret(SVector{3, Float32}, reshape(shs, :, n)),
        Val(sh_degree); ndrange=n)

    count_tiles_per_gaussian!(kab, Int(BLOCK_SIZE))(
        # Output.
        rast.gstate.tiles_touched,
        # Input.
        rast.gstate.means_2d,
        rast.gstate.radii,
        rast.grid, BLOCK; ndrange=n)

    cumsum!(rast.gstate.points_offset, rast.gstate.tiles_touched)
    # Get total number of tiles touched.
    n_rendered = Int(@allowscalar rast.gstate.points_offset[end])
    n_rendered == 0 && return rast.image

    length(rast.bstate) == n_rendered ||
        (rast.bstate = BinningState(kab, n_rendered))

    # For each instance to be rendered, produce [tile | depth] key
    # and corresponding duplicated Gaussian indices to be sorted.
    duplicate_with_keys!(kab, Int(BLOCK_SIZE))(
        # Output.
        rast.bstate.gaussian_keys_unsorted,
        rast.bstate.gaussian_values_unsorted,
        # Input.
        rast.gstate.means_2d,
        rast.gstate.depths,
        rast.gstate.points_offset,
        rast.gstate.radii, rast.grid, BLOCK; ndrange=n)

    sortperm!(rast.bstate.permutation, rast.bstate.gaussian_keys_unsorted)
    _permute!(kab)(
        rast.bstate.gaussian_keys_sorted, rast.bstate.gaussian_keys_unsorted,
        rast.bstate.permutation; ndrange=n_rendered)
    _permute!(kab)(
        rast.bstate.gaussian_values_sorted, rast.bstate.gaussian_values_unsorted,
        rast.bstate.permutation; ndrange=n_rendered)

    # Identify start-end of per-tile workloads in sorted keys.
    fill!(rast.istate.ranges, 0u32)
    if n_rendered > 0
        identify_tile_range!(kab, Int(BLOCK_SIZE))(
            rast.istate.ranges, rast.bstate.gaussian_keys_sorted;
            ndrange=n_rendered)
    end

    render!(kab, (Int.(BLOCK)...,), (width, height))(
        # Outputs.
        rast.image,
        rast.istate.n_contrib,
        rast.istate.accum_α,
        # Inputs.
        rast.bstate.gaussian_values_sorted,
        rast.gstate.means_2d,
        opacities,
        rast.gstate.conic_opacities,
        rast.gstate.rgbs,
        rast.gstate.depths,

        rast.istate.ranges,
        SVector{2, Int32}(width, height),
        background,
        BLOCK, Val(BLOCK_SIZE), Val(3i32))

    return rast.image
end

function ∇rasterize(
    vpixels::AbstractArray{Float32, 3},
    means_3d::AbstractMatrix{Float32},
    shs::AbstractArray{Float32, 3},
    scales::AbstractMatrix{Float32},
    rotations::AbstractMatrix{Float32},
    opacities::AbstractMatrix{Float32},
    radii::AbstractVector{Int32};
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
    background::SVector{3, Float32},
)
    kab = get_backend(rast)
    n = size(means_3d, 2)

    (; width, height) = resolution(camera)
    @assert width % 16 == 0 && height % 16 == 0

    vcolors = KA.zeros(kab, Float32, (3, n))
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

    ∇render!(kab, (Int.(BLOCK)...,), (width, height))(
        # Outputs.
        vcolors,
        vopacities,
        vconics,
        reshape(reinterpret(Float32, rast.gstate.∇means_2d), 2, :),
        # Inputs.
        reshape(reinterpret(SVector{3, Float32}, vpixels), size(vpixels)[2:3]),
        rast.istate.n_contrib,
        rast.istate.accum_α,

        rast.bstate.gaussian_values_sorted,
        rast.gstate.means_2d,
        opacities,
        rast.gstate.conic_opacities,
        rast.gstate.rgbs,

        rast.istate.ranges,
        SVector{2, Int32}(width, height), background,
        rast.grid, BLOCK, Val(BLOCK_SIZE), Val(3i32))

    _as_T(T, x) = reinterpret(T, reshape(x, :))

    R_w2c = SMatrix{3, 3, Float32}(camera.w2c[1:3, 1:3])
    t_w2c = SVector{3, Float32}(camera.w2c[1:3, 4])
    ∇project!(kab, Int(BLOCK_SIZE))(
        # Output.
        _as_T(SVector{3, Float32}, vmeans),
        _as_T(SVector{3, Float32}, vscales),
        _as_T(SVector{4, Float32}, vrot),

        # Input grad outputs.
        rast.gstate.∇means_2d,
        _as_T(SVector{3, Float32}, vconics),

        rast.gstate.conic_opacities,
        rast.gstate.radii,
        # Input Gaussians.
        _as_T(SVector{3, Float32}, means_3d),
        _as_T(SVector{3, Float32}, scales),
        _as_T(SVector{4, Float32}, rotations),
        # Input camera properties.
        R_w2c, t_w2c, K.focal, Int32.(K.resolution), K.principal; ndrange=n)

    ∇spherical_harmonics!(kab, Int(BLOCK_SIZE))(
        # Output.
        reinterpret(SVector{3, Float32}, reshape(vshs, :, n)),
        _as_T(SVector{3, Float32}, vmeans),
        # Input.
        _as_T(SVector{3, Float32}, means_3d),
        reinterpret(SVector{3, Float32}, reshape(shs, :, n)),
        rast.gstate.clamped,
        _as_T(SVector{3, Float32}, vcolors),
        camera.camera_center,
        Val(sh_degree); ndrange=n)

    return vmeans, vshs, vopacities, vscales, vrot
end

function ChainRulesCore.rrule(::typeof(rasterize),
    means_3d, shs, opacities, scales, rotations;
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
    background::SVector{3, Float32},
)
    image = rasterize(
        means_3d, shs, opacities, scales, rotations;
        rast, camera, sh_degree, background)

    function _pullback(vpixels)
        ∇ = ∇rasterize(
            vpixels, means_3d, shs, scales, rotations, opacities,
            rast.gstate.radii; rast, camera, sh_degree, background)
        return (NoTangent(), ∇...)
    end
    return image, _pullback
end
