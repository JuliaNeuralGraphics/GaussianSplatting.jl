# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
sdiagm(x, y, z) = SMatrix{3, 3, Float32, 9}(
    x, 0f0, 0f0,
    0f0, y, 0f0,
    0f0, 0f0, z)

@inbounds smat3f0(x) = SMatrix{3, 3, Float32, 9}(
    x[1, 1], x[2, 1], x[3, 1],
    x[1, 2], x[2, 2], x[3, 2],
    x[1, 3], x[2, 3], x[3, 3])

@inbounds svec3f0(x) = SVector{3, Float32}(x[1], x[2], x[3])

gpu_floor(T, x) = unsafe_trunc(T, floor(x))
gpu_ceil(T, x) = unsafe_trunc(T, ceil(x))
gpu_cld(x::X, y::T) where {X, T} = unsafe_trunc(T, floor(Float32(x + y - one(X)) / Float32(y)))

"""
    ellipse_radius_sq(opacity::Float32)

Squared Mahalanobis radius of the `α ≥ 1/255` contour of a Gaussian.

`render!` blends a fragment iff `α = min(0.99f0, opacity * exp(-σ)) ≥ 1/255`, and
since `0.99 > 1/255` the clamp never decides the test, so the fragment survives iff
`σ ≤ log(255·opacity)`, i.e. iff `δᵀ·Σ⁻¹·δ ≤ 2·log(255·opacity)` (note `2σ` is the
quadratic form `δᵀ·Σ⁻¹·δ`). The result is `≤ 0` exactly when `opacity ≤ 1/255`,
i.e. when the Gaussian can never clear the blend threshold anywhere.
"""
@inline ellipse_radius_sq(opacity::Float32) = 2f0 * log(255f0 * opacity)

"""
    ellipse_extent(conic, radius_sq)

Per-axis half-extents of the ellipse `δᵀ·conic·δ ≤ radius_sq`, from the marginal
variances `Σ[1, 1] = conic[3] / detᶜ` and `Σ[2, 2] = conic[1] / detᶜ`.

Tighter than a disc of radius `sqrt(radius_sq · λ_max)` for anything anisotropic.
"""
@inline function ellipse_extent(conic::SVector{3, Float32}, radius_sq::Float32)
    detᶜ = max(conic[1] * conic[3] - conic[2]^2, 1f-20)
    r = sqrt(radius_sq)
    return SVector{2, Float32}(
        r * sqrt(conic[3] / detᶜ),
        r * sqrt(conic[1] / detᶜ))
end

"""
    sample_tile_span(c_lo, c_hi, block, lo, hi)

Half-open tile span `[a, b)` covering every *integer* pixel sample in the
continuous interval `[c_lo, c_hi]`, clamped to `[lo, hi]`.

`render!` samples at integer pixel coordinates (`δ = xy .- pix` with 0-based
integer `pix`), so the covered samples are `ceil(c_lo):floor(c_hi)` — being
sample-exact rather than merely conservative is the whole point of the exact
intersection. Returns an empty span when no sample falls inside.
"""
@inline function sample_tile_span(
    c_lo::Float32, c_hi::Float32, block::Int32, lo::Int32, hi::Int32,
)
    # Clamp in float space before truncating: the extent can be arbitrarily
    # large and `unsafe_trunc` on an out-of-range float is undefined.
    limit = Float32(hi) + 1f0
    t_lo = clamp(ceil(c_lo) / block, -1f0, limit)
    t_hi = clamp(floor(c_hi) / block, -1f0, limit)
    a = clamp(gpu_floor(Int32, t_lo), lo, hi)
    b = clamp(gpu_floor(Int32, t_hi) + 1i32, lo, hi)
    # `b < a` can only happen when the interval holds no sample at all.
    return b < a ? (lo, lo) : (a, b)
end

"""
    ellipse_range_bound(conic, radius_sq, y0, y1)

Closed-form x-span of `a·x² + 2b·x·y + c·y² ≤ radius_sq` (with
`(a, b, c) = conic`) restricted to the strip `y ∈ [y0, y1]`.

Port of LichtFeld-Studio's `ellipse_range_bound`
(`fastgs/rasterization/include/kernel_utils.cuh`), itself from StopThePop;
derivation: https://www.desmos.com/calculator/sjdw2xmohr.
"""
@inline function ellipse_range_bound(
    conic::SVector{3, Float32}, radius_sq::Float32, y0::Float32, y1::Float32,
)
    a, b, c = conic[1], conic[2], conic[3]
    det = max(a * c - b * b, 1f-20)
    # `±ym` are the `y` at which the ellipse is widest in `x`.
    ym = -b / c * sqrt(max(c * radius_sq / det, 0f0))
    v0, v1 = clamp(-ym, y0, y1), clamp(ym, y0, y1)
    bv0, bv1 = -b * v0, -b * v1
    inv_a = 1f0 / a
    x0 = inv_a * (bv0 - sqrt(max(bv0 * bv0 - a * (c * v0 * v0 - radius_sq), 0f0)))
    x1 = inv_a * (bv1 + sqrt(max(bv1 * bv1 - a * (c * v1 * v1 - radius_sq), 0f0)))
    return SVector{2, Float32}(x0, x1)
end

"""
    ellipse_tile_bounds(mean_2d, conic, opacity, grid, block)

Coarse, opacity-aware, half-open tile AABB of the ellipse a Gaussian actually
covers — the AABB of the *ellipse*, not of a disc of radius `3σ`.

Returns `(rmin, rmax, radius_sq)`; `radius_sq ≤ 0` (or an empty rect) means the
Gaussian touches no tile. `ellipse_tile_span` then refines this rect row by row.
"""
@inline function ellipse_tile_bounds(
    mean_2d::SVector{2, Float32}, conic::SVector{3, Float32}, opacity::Float32,
    grid::SVector{2, Int32}, block::SVector{2, Int32},
)
    radius_sq = ellipse_radius_sq(opacity)
    z = zeros(SVector{2, Int32})
    radius_sq > 0f0 || return z, z, 0f0

    extent = ellipse_extent(conic, radius_sq)
    xlo, xhi = sample_tile_span(
        mean_2d[1] - extent[1], mean_2d[1] + extent[1], block[1], 0i32, grid[1])
    ylo, yhi = sample_tile_span(
        mean_2d[2] - extent[2], mean_2d[2] + extent[2], block[2], 0i32, grid[2])
    return (
        SVector{2, Int32}(xlo, ylo),
        SVector{2, Int32}(xhi, yhi),
        radius_sq)
end

"""
    ellipse_tile_span(mean_2d, conic, radius_sq, s, lo, hi, block, Val(transposed))

Exact half-open tile span of the ellipse within one tile row (`transposed=false`:
`s` is a 0-based `tile_y`, the result is an x-span) or one tile column
(`transposed=true`: `s` is a 0-based `tile_x`, the result is a y-span).

Scanning the *shorter* axis of the coarse rect keeps the loop short in either
orientation; the caller picks `transposed` from the rect alone so that the
counting and the key-emitting walks always take the same branch.

The strip is `[s·block, s·block + block - 1]` — the Gaussian's *sample* rows, one
short of the continuous tile rectangle, because `render!` only evaluates integer
pixel coordinates.
"""
@inline function ellipse_tile_span(
    mean_2d::SVector{2, Float32}, conic::SVector{3, Float32}, radius_sq::Float32,
    s::Int32, lo::Int32, hi::Int32, block::SVector{2, Int32}, ::Val{transposed},
) where {transposed}
    # `ellipse_range_bound` solves for `x` given a `y`-strip. To scan columns
    # instead, swap the axes and transpose the quadratic form to `(c, b, a)`.
    conicᵀ, m_major, m_minor, b_major, b_minor = if transposed
        SVector{3, Float32}(conic[3], conic[2], conic[1]),
        mean_2d[1], mean_2d[2], block[1], block[2]
    else
        conic, mean_2d[2], mean_2d[1], block[2], block[1]
    end

    # `ellipse_range_bound` works in `pixel - mean` space; the form is even, so
    # the sign relative to `render!`'s `δ = mean - pixel` does not matter.
    y0 = Float32(s * b_major) - m_major
    y1 = y0 + Float32(b_major - 1i32)
    bound = ellipse_range_bound(conicᵀ, radius_sq, y0, y1)
    return sample_tile_span(
        bound[1] + m_minor, bound[2] + m_minor, b_minor, lo, hi)
end

"""
Number of tiles the Gaussian's `α ≥ 1/255` ellipse covers. Shares
`ellipse_tile_span` with `_ellipse_walk_emit!` so the two walks cannot drift.
"""
@inline function _ellipse_walk_count(
    mean_2d::SVector{2, Float32}, conic::SVector{3, Float32}, radius_sq::Float32,
    rmin::SVector{2, Int32}, rmax::SVector{2, Int32}, block::SVector{2, Int32},
    ::Val{transposed},
) where {transposed}
    major_lo, major_hi = transposed ? (rmin[1], rmax[1]) : (rmin[2], rmax[2])
    minor_lo, minor_hi = transposed ? (rmin[2], rmax[2]) : (rmin[1], rmax[1])

    n = 0i32
    for s in major_lo:(major_hi - 1i32)
        lo, hi = ellipse_tile_span(
            mean_2d, conic, radius_sq, s, minor_lo, minor_hi, block, Val(transposed))
        n += hi - lo
    end
    return n
end

"""
Emit one `[tile_id | depth]` key per covered tile, starting at `offset` and never
writing past `offset_end`. Returns the next free offset.
"""
@inline function _ellipse_walk_emit!(
    gaussian_keys::AbstractVector{UInt64}, gaussian_values::AbstractVector{UInt32},
    i, depth::UInt64,
    mean_2d::SVector{2, Float32}, conic::SVector{3, Float32}, radius_sq::Float32,
    rmin::SVector{2, Int32}, rmax::SVector{2, Int32},
    grid::SVector{2, Int32}, block::SVector{2, Int32},
    offset::Int32, offset_end::Int32, ::Val{transposed},
) where {transposed}
    major_lo, major_hi = transposed ? (rmin[1], rmax[1]) : (rmin[2], rmax[2])
    minor_lo, minor_hi = transposed ? (rmin[2], rmax[2]) : (rmin[1], rmax[1])

    for s in major_lo:(major_hi - 1i32)
        lo, hi = ellipse_tile_span(
            mean_2d, conic, radius_sq, s, minor_lo, minor_hi, block, Val(transposed))
        for m in lo:(hi - 1i32)
            # `_ellipse_walk_count` produced `offset_end`, so this can only trip
            # if the two walks disagreed; bail out rather than corrupt a
            # neighbouring Gaussian's slots.
            offset > offset_end && return offset

            x, y = transposed ? (s, m) : (m, s)
            key::UInt64 = UInt64(y) * grid[1] + x
            key <<= 32
            key |= depth
            gaussian_keys[offset] = key
            gaussian_values[offset] = i
            offset += 1i32
        end
    end
    return offset
end

# Spherical harmonics coefficients up to a 3rd degree.

const SH0::Float32 = 0.28209479177387814f0
const SH1::Float32 = 0.4886025119029199f0

const SH2C1::Float32 =  1.0925484305920792f0
const SH2C2::Float32 = -1.0925484305920792f0
const SH2C3::Float32 =  0.31539156525252005f0
const SH2C4::Float32 = -1.0925484305920792f0
const SH2C5::Float32 =  0.5462742152960396f0

const SH3C1::Float32 = -0.5900435899266435f0
const SH3C2::Float32 =  2.890611442640554f0
const SH3C3::Float32 = -0.4570457994644658f0
const SH3C4::Float32 =  0.3731763325901154f0
const SH3C5::Float32 = -0.4570457994644658f0
const SH3C6::Float32 =  1.445305721320277f0
const SH3C7::Float32 = -0.5900435899266435f0

"""
For each tile in `ranges`, given a sorted list of keys,
find start/end index ranges.
I.e. tile 0 spans gaussian keys from `1` to `k₁` index,
tile 1 from `k₁` to `k₂`, etc.
"""
@kernel cpu=false inbounds=true function identify_tile_range!(
    ranges::AbstractMatrix{UInt32},
    gaussian_keys::AbstractVector{UInt64},
)
    n = @ndrange()[1]
    i = @index(Global)

    tile = (gaussian_keys[i] >> 32) + 1u32

    if i == 1
        ranges[1, tile] = 0u32
    else
        prev_tile = (gaussian_keys[i - 1] >> 32) + 1u32
        if tile != prev_tile
            ranges[2, prev_tile] = i - 1u32
            ranges[1, tile] = i - 1u32
        end
    end

    if i == n
        ranges[2, tile] = n
    end
end

@kernel cpu=false inbounds=true function _permute!(y, x, ix)
    i = @index(Global)
    y[i] = x[ix[i]]
end

@kernel cpu=false inbounds=true function duplicate_with_keys!(
    # Outputs.
    gaussian_keys::AbstractVector{UInt64},
    gaussian_values::AbstractVector{UInt32},
    # Inputs.
    means_2d::AbstractVector{SVector{2, Float32}},
    conics::AbstractVector{SVector{3, Float32}},
    opacities::AbstractMatrix{Float32},
    depths::AbstractVector{Float32},
    gaussian_offset::AbstractVector{Int32},
    radii::AbstractVector{Int32},
    grid::SVector{2, Int32}, block::SVector{2, Int32},
)
    i = @index(Global)

    # No key/value for invisible Gaussians.
    # No need for the default key/value, since `gaussian_offset` covers
    # only valid gaussians.
    radii[i] > 0i32 || return

    offset = i == 1 ? 1i32 : (gaussian_offset[i - 1] + 1i32)
    offset_end = gaussian_offset[i]
    offset ≤ offset_end || return

    # For each tile the `α ≥ 1/255` ellipse covers, emit a key/value pair.
    # Key: [tile_id | depth], value: id of the Gaussian.
    # Sorting the values with this key yields Gaussian ids in a list,
    # such that they are first sorted by the tile and then depth.
    depth::UInt64 = reinterpret(UInt32, depths[i])

    mean_2d, conic = means_2d[i], conics[i]
    rmin, rmax, radius_sq = ellipse_tile_bounds(
        mean_2d, conic, opacities[i], grid, block)

    if radius_sq > 0f0
        # Scan the shorter axis. Derived from the rect alone, so this matches
        # the branch `count_tiles_per_gaussian!` took.
        offset = (rmax[2] - rmin[2]) > (rmax[1] - rmin[1]) ?
            _ellipse_walk_emit!(
                gaussian_keys, gaussian_values, i, depth,
                mean_2d, conic, radius_sq, rmin, rmax, grid, block,
                offset, offset_end, Val(true)) :
            _ellipse_walk_emit!(
                gaussian_keys, gaussian_values, i, depth,
                mean_2d, conic, radius_sq, rmin, rmax, grid, block,
                offset, offset_end, Val(false))
    end

    # Defense in depth: if the walk emitted fewer keys than
    # `count_tiles_per_gaussian!` counted, the leftover slots would still hold
    # keys from a previous frame, whose tile ids can index `ranges` out of
    # bounds. Pad with a sentinel tile one past the last real one — it sorts to
    # the very end and neither `render!` nor `∇render!` ever visits it (they
    # index `ranges` only up to `prod(grid)`, and `ImageState` allocates one
    # extra column for exactly this).
    sentinel::UInt64 = (UInt64(grid[1]) * UInt64(grid[2])) << 32
    while offset ≤ offset_end
        gaussian_keys[offset] = sentinel
        gaussian_values[offset] = i
        offset += 1i32
    end
end

@kernel cpu=false inbounds=true function count_tiles_per_gaussian!(
    # Output.
    tiles_touched::AbstractVector{Int32},
    # Input.
    means_2d::AbstractVector{SVector{2, Float32}},
    conics::AbstractVector{SVector{3, Float32}},
    opacities::AbstractMatrix{Float32},
    radii::AbstractVector{Int32},
    tile_grid::SVector{2, Int32},
    tile_size::SVector{2, Int32},
)
    i = @index(Global)
    if !(radii[i] > 0i32)
        tiles_touched[i] = 0i32
        return
    end

    mean_2d, conic = means_2d[i], conics[i]
    rmin, rmax, radius_sq = ellipse_tile_bounds(
        mean_2d, conic, opacities[i], tile_grid, tile_size)
    if !(radius_sq > 0f0)
        tiles_touched[i] = 0i32
        return
    end

    # Scan the shorter axis; `duplicate_with_keys!` repeats this choice.
    tiles_touched[i] = (rmax[2] - rmin[2]) > (rmax[1] - rmin[1]) ?
        _ellipse_walk_count(
            mean_2d, conic, radius_sq, rmin, rmax, tile_size, Val(true)) :
        _ellipse_walk_count(
            mean_2d, conic, radius_sq, rmin, rmax, tile_size, Val(false))
end
