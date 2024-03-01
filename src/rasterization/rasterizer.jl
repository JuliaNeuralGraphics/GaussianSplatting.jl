include("states.jl")
include("utils.jl")
include("forward.jl")
include("backward.jl")

mutable struct GaussianRasterizer{
    I <: ImageState,
    G <: GeometryState,
    B <: BinningState,
    K <: AbstractArray{Float32, 3},
    A,
}
    istate::I
    gstate::G
    bstate::B

    image::K

    # If not nothing, then it is a buffer of the same resolution as `image`.
    # 1st channel - depth
    # 2nd channel - silhouette:
    #   0 - if nothing is rendered in the pixel
    #   1 - if pixel if fully opaque
    auxiliary::A

    grid::SVector{2, Int32}
end

function GaussianRasterizer(kab, camera::Camera; auxiliary::Bool = false)
    (; width, height) = resolution(camera)
    @assert width % 16 == 0 && height % 16 == 0

    grid = SVector{2, Int32}(cld(width, BLOCK[1]), cld(height, BLOCK[2]))
    istate = ImageState(kab; width, height, grid_size=Int(prod(grid)))
    gstate = GeometryState(kab, 0)
    bstate = BinningState(kab, 0)

    image = KA.allocate(kab, Float32, (3, width, height))
    aux = auxiliary ? KA.allocate(kab, Float32, (2, width, height)) : nothing
    GaussianRasterizer(istate, gstate, bstate, image, aux, grid)
end

KernelAbstractions.get_backend(r::GaussianRasterizer) = get_backend(r.image)

has_auxiliary(r::GaussianRasterizer) = r.auxiliary ≢ nothing

function to_raw_depth(r::GaussianRasterizer; normalize::Bool = false)
    isnothing(r.auxiliary) && error("""
    GaussianRasterizer does not have auxiliary information.
    To compute it, re-create rasterizer with `auxiliary=true` keyword argument.
    """)

    raw_aux = Array(r.auxiliary)
    depth = raw_aux[1, :, :]
    normalize || return depth

    # Normalize by max depth to be in [0, 1] range.
    max_depth = 0f0
    for d in depth
        isinf(d) && continue
        max_depth = max(max_depth, d)
    end
    depth .*= 1f0 / (max_depth + 1f-6)

    return depth
end

# OpenGL convertions.

function gl_texture(r::GaussianRasterizer)
    @show size(r.image)
    raw_img = clamp01!(Array(r.image)[:, :, end:-1:1])
    return colorview(RGB, raw_img)
end

function to_gl_depth(r::GaussianRasterizer)
    depth = to_raw_depth(r; normalize=true)
    depth = reshape(depth[:, end:-1:1], 1, size(depth)...)
    return colorview(Gray, depth)
end

function to_gl_uncertainty(r::GaussianRasterizer)
    raw_aux = Array(r.auxiliary)
    uncertainty = reshape(raw_aux[2, :, end:-1:1], 1, size(raw_aux)[2:3]...)
    return colorview(Gray, uncertainty)
end

# Image conversions.

function to_image(r::GaussianRasterizer)
    raw_img = clamp01!(permutedims(Array(r.image), (1, 3, 2)))
    return colorview(RGB, raw_img)
end

function to_depth(r::GaussianRasterizer; normalize::Bool = false)
    depth = to_raw_depth(r; normalize)
    depth = transpose(depth)
    return colorview(Gray, depth)
end

function to_uncertainty(r::GaussianRasterizer)
    raw_aux = Array(r.auxiliary)
    uncertainty = transpose(raw_aux[2, :, :])
    return colorview(Gray, uncertainty)
end

function (rast::GaussianRasterizer)(
    means_3d, opacities, scales, rotations, shs;
    camera::Camera, sh_degree::Int,
    background::SVector{3, Float32} = zeros(SVector{3, Float32}),
    covisibility::Maybe{AbstractVector{Bool}} = nothing,
)
    # Apply activation functions and rasterize.
    rasterize(
        means_3d,
        shs,
        NU.sigmoid.(opacities),
        exp.(scales),
        rotations ./ sqrt.(sum(abs2, rotations; dims=1));
        rast, sh_degree, camera, background, covisibility)
end

# Supports `q`, `t` gradient.
function (rast::GaussianRasterizer)(
    means_3d, opacities, scales, rotations, shs,
    q, t;
    camera::Camera, sh_degree::Int,
    background::SVector{3, Float32} = zeros(SVector{3, Float32}),
    covisibility::Maybe{AbstractVector{Bool}} = nothing,
)
    @assert sh_degree == 0
    @assert size(q) == (4,)
    @assert size(t) == (3,)

    # Normalize quaterion, convert to 3x3 matrix and apply to gaussians.
    nq = q ./ sqrt.(sum(abs2, q))
    r2c = quat2mat(nq)
    local_means_3d = r2c * means_3d .+ t

    normalized_rotations = rotations ./ sqrt.(sum(abs2, rotations; dims=1))
    local_rotations = quat_mul(nq, normalized_rotations)

    # Construct camera with identity w2c.
    id_camera = @ignore_derivatives(Camera(
        SMatrix{3, 3, Float32}(I), zeros(SVector{3, Float32});
        camera.intrinsics, camera.img_name))

    rasterize(
        local_means_3d,
        shs,
        NU.sigmoid.(opacities),
        exp.(scales),
        local_rotations ./ sqrt.(sum(abs2, local_rotations; dims=1));
        rast, sh_degree, camera=id_camera, background, covisibility)
end

"""
Computing gradients w.r.t. means_3d, shs, opacities, scales, rotations?
"""
function rasterize(
    means_3d, shs, opacities, scales, rotations;
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
    background::SVector{3, Float32},
    covisibility::Maybe{AbstractVector{Bool}},
)
    kab = get_backend(rast)
    n = size(means_3d, 2)
    if length(rast.gstate) != n
        rast.gstate = GeometryState(kab, n)
    end

    (; width, height) = resolution(camera)
    @assert width % 16 == 0 && height % 16 == 0

    K = camera.intrinsics
    tan_fov_xy = tan.(deg2rad.(0.5f0 .* NU.focal2fov.(K.resolution, K.focal)))
    focal_xy = K.resolution ./ (2f0 .* tan_fov_xy)

    fill!(rast.image, 0f0)
    has_auxiliary(rast) && fill!(rast.auxiliary, 0f0)

    _as_T(T, x) = reinterpret(T, reshape(x, :))
    scale_modifier = 1f0 # TODO compute correctly

    # Preprocess per-Gaussian: transformation, bounding, sh-to-rgb.
    _preprocess!(kab, Int(BLOCK_SIZE))(
        # Output.
        rast.gstate.cov3Ds,
        rast.gstate.depths,
        rast.gstate.radii,
        rast.gstate.means_2d,
        rast.gstate.conic_opacities,
        rast.gstate.tiles_touched,
        rast.gstate.rgbs,
        rast.gstate.clamped,
        # Input.
        _as_T(SVector{3, Float32}, means_3d),
        _as_T(SVector{3, Float32}, scales),
        _as_T(SVector{4, Float32}, rotations),
        reinterpret(SVector{3, Float32}, reshape(shs, :, n)), Val(sh_degree),
        opacities,
        camera.full_projection,
        camera.w2c,
        camera.camera_center,
        SVector{2, Int32}(width, height),
        rast.grid, BLOCK, focal_xy, tan_fov_xy, scale_modifier; ndrange=n)

    cumsum!(rast.gstate.points_offset, rast.gstate.tiles_touched)
    # Get total number of tiles touched.
    n_rendered::Int = Array(@view(rast.gstate.points_offset[end]))[1]
    n_rendered == 0 && return rast.image

    if length(rast.bstate) != n_rendered # TODO optimize
        rast.bstate = BinningState(kab, n_rendered)
    end

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
        rast.auxiliary,
        covisibility,
        rast.istate.n_contrib,
        rast.istate.accum_α,
        # Inputs.
        rast.bstate.gaussian_values_sorted,
        rast.gstate.means_2d,
        rast.gstate.conic_opacities,
        rast.gstate.rgbs,
        rast.gstate.depths,

        rast.istate.ranges,
        SVector{2, Int32}(width, height),
        background,
        BLOCK, Val(BLOCK_SIZE), Val(3i32))

    return rast.image
end

# NOTE: storing radii, means_2d, ∇means_2d in rast.gstate
function ∇rasterize(
    ∂L∂pixels::AbstractArray{Float32, 3},
    means_3d, shs, scales, rotations,
    radii::AbstractVector{Int32};
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
    background::SVector{3, Float32},
)
    kab = get_backend(rast)
    n = size(means_3d, 2)

    (; width, height) = resolution(camera)
    @assert width % 16 == 0 && height % 16 == 0

    # Auxiliary buffers.
    ∂L∂colors = KA.zeros(kab, Float32, (3, n))
    ∂L∂conic_opacities = KA.zeros(kab, Float32, (2, 2, n))
    ∂L∂cov = KA.zeros(kab, Float32, (6, n))

    # TODO pre-allocate
    ∂L∂means = KA.zeros(kab, Float32, (3, n))
    ∂L∂shs = KA.zeros(kab, Float32, size(shs))
    ∂L∂opacities = KA.zeros(kab, Float32, (1, n))
    ∂L∂scales = KA.zeros(kab, Float32, (3, n))
    ∂L∂rot = KA.zeros(kab, Float32, (4, n))
    fill!(reinterpret(Float32, rast.gstate.∇means_2d), 0f0)

    K = camera.intrinsics
    tan_fov_xy = tan.(deg2rad.(0.5f0 .* NU.focal2fov.(K.resolution, K.focal)))
    focal_xy = K.resolution ./ (2f0 .* tan_fov_xy)

    ∇render!(kab, (Int.(BLOCK)...,), (width, height))(
        # Outputs.
        ∂L∂colors,
        ∂L∂opacities,
        ∂L∂conic_opacities,
        reshape(reinterpret(Float32, rast.gstate.∇means_2d), 2, :),
        # Inputs.
        reshape(reinterpret(SVector{3, Float32}, ∂L∂pixels), size(∂L∂pixels)[2:3]),
        # (output from the forward `render!` pass)
        rast.istate.n_contrib,
        rast.istate.accum_α,

        rast.bstate.gaussian_values_sorted,
        rast.gstate.means_2d,
        rast.gstate.conic_opacities,
        rast.gstate.rgbs,

        rast.istate.ranges,
        SVector{2, Int32}(width, height), background,
        rast.grid, BLOCK, Val(BLOCK_SIZE), Val(3i32))

    _as_T(T, x) = reinterpret(T, reshape(x, :))
    ∇compute_cov_2d!(kab, Int(BLOCK_SIZE))(
        # Outputs.
        _as_T(SVector{3, Float32}, ∂L∂means),
        ∂L∂cov,
        # Inputs.
        ∂L∂conic_opacities,
        rast.gstate.cov3Ds,
        radii,
        _as_T(SVector{3, Float32}, means_3d),
        camera.w2c,
        focal_xy, tan_fov_xy; ndrange=n)

    scale_modifier = 1f0
    ∇_preprocess!(kab, Int(BLOCK_SIZE))(
        # Outputs.
        _as_T(SVector{3, Float32}, ∂L∂means),
        reinterpret(SVector{3, Float32}, reshape(∂L∂shs, :, n)),
        _as_T(SVector{3, Float32}, ∂L∂scales),
        _as_T(SVector{4, Float32}, ∂L∂rot),
        # Inputs.
        ∂L∂cov,
        _as_T(SVector{3, Float32}, ∂L∂colors),
        rast.gstate.∇means_2d,
        radii,
        _as_T(SVector{3, Float32}, means_3d),
        _as_T(SVector{3, Float32}, scales),
        _as_T(SVector{4, Float32}, rotations),
        reinterpret(SVector{3, Float32}, reshape(shs, :, n)), Val(sh_degree),
        rast.gstate.clamped,
        camera.full_projection,
        camera.camera_center,
        scale_modifier; ndrange=n)

    return (∂L∂means, ∂L∂shs, ∂L∂opacities, ∂L∂scales, ∂L∂rot)
end

function ChainRulesCore.rrule(
    ::typeof(rasterize), means_3d, shs, opacities, scales, rotations;
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
    background::SVector{3, Float32},
    covisibility::Maybe{AbstractVector{Bool}},
)
    image = rasterize(
        means_3d, shs, opacities, scales, rotations;
        rast, camera, sh_degree, background, covisibility)

    function _rasterize_pullback(∂L∂pixels)
        ∇ = ∇rasterize(
            ∂L∂pixels, means_3d, shs, scales, rotations,
            rast.gstate.radii; rast, camera, sh_degree, background)
        return (NoTangent(), ∇...)
    end
    return image, _rasterize_pullback
end

"""
For each tile in `ranges`, given a sorted list of keys,
find start/end index ranges.
I.e. tile 0 spans gaussian keys from `1` to `k₁` index,
tile 1 from `k₁` to `k₂`, etc.
"""
@kernel function identify_tile_range!(
    ranges::AbstractMatrix{UInt32},
    gaussian_keys::AbstractVector{UInt64},
)
    @uniform n = @ndrange()[1]
    i = @index(Global)

    @inbounds tile = (gaussian_keys[i] >> 32) + 1u32

    if i == 1
        @inbounds ranges[1, tile] = 0u32
    else
        @inbounds prev_tile = (gaussian_keys[i - 1] >> 32) + 1u32
        if tile != prev_tile
            @inbounds ranges[2, prev_tile] = i - 1u32
            @inbounds ranges[1, tile] = i - 1u32
        end
    end

    if i == n
        @inbounds ranges[2, tile] = n
    end
end

@kernel function _permute!(y, @Const(x), @Const(ix))
    i = @index(Global)
    @inbounds y[i] = x[ix[i]]
end

@kernel function duplicate_with_keys!(
    # Outputs.
    gaussian_keys::AbstractVector{UInt64},
    gaussian_values::AbstractVector{UInt32},
    # Inputs.
    means_2d::AbstractVector{SVector{2, Float32}},
    depths::AbstractVector{Float32},
    gaussian_offset::AbstractVector{Int32},
    radii::AbstractVector{Int32},
    grid::SVector{2, Int32}, block::SVector{2, Int32},
)
    i = @index(Global)

    # No key/value for invisible Gaussians.
    # No need for the default key/value, since `gaussian_offset` covers
    # only valid gaussians.
    @inbounds radius = radii[i]
    if radius > 0
        @inbounds rmin, rmax = get_rect(means_2d[i], radius, grid, block)
        # For each tile that the bounding rect overlaps, emit a key/value pair.
        # Key: [tile_id | depth], value: id of the Gaussian.
        # Sorting the values with this key yields Gaussian ids in a list,
        # such that they are first sorted by the tile and then depth.
        @inbounds depth::UInt64 = reinterpret(UInt32, depths[i])

        @inbounds offset = i == 1 ? 1i32 : (gaussian_offset[i - 1] + 1i32)
        for y in rmin[2]:(rmax[2] - 1i32), x in rmin[1]:(rmax[1] - 1i32)
            key::UInt64 = UInt64(y) * grid[1] + x
            key <<= 32
            key |= depth
            @inbounds gaussian_keys[offset] = key
            @inbounds gaussian_values[offset] = i
            offset += 1
        end
    end
end
