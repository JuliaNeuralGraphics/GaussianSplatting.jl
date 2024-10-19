# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
function in_frustum(point::SVector{4, Float32}, view::SMatrix{4, 4, Float32, 16})
    depth = (view * point)[3]
    return depth > 0.2f0, depth
end

"""
`q` must be normalized.
"""
function quat2mat(q::SVector{4, Float32})
    r, x, y, z = q
    SMatrix{3, 3, Float32, 9}(
        1f0 - 2f0 * (y^2 + z^2), 2f0 * (x * y + r * z), 2f0 * (x * z - r * y),
        2f0 * (x * y - r * z), 1f0 - 2f0 * (x^2 + z^2), 2f0 * (y * z + r * x),
        2f0 * (x * z + r * y), 2f0 * (y * z - r * x), 1f0 - 2f0 * (x^2 + y^2))
end

function quat2mat(q)
    r, x, y, z = q[[1]], q[[2]], q[[3]], q[[4]]
    reshape(vcat(
        @.(1f0 - 2f0 * (y^2 + z^2)),
        @.(2f0 * (x * y + r * z)),
        @.(2f0 * (x * z - r * y)),
        @.(2f0 * (x * y - r * z)),
        @.(1f0 - 2f0 * (x^2 + z^2)),
        @.(2f0 * (y * z + r * x)),
        @.(2f0 * (x * z + r * y)),
        @.(2f0 * (y * z - r * x)),
        @.(1f0 - 2f0 * (x^2 + y^2))), 3, 3)
end

function quat_mul(q1, q2)
    @assert ndims(q1) == 1
    @assert ndims(q2) == 2

    r1, x1, y1, z1 = q1[[1]], q1[[2]], q1[[3]], q1[[4]]
    r2, x2, y2, z2 = q2[[1], :], q2[[2], :], q2[[3], :], q2[[4], :]

    r = @. r1 * r2 - x1 * x2 - y1 * y2 - z1 * z2
    x = @. r1 * x2 + x1 * r2 + y1 * z2 - z1 * y2
    y = @. r1 * y2 - x1 * z2 + y1 * r2 + z1 * x2
    z = @. r1 * z2 + x1 * y2 - y1 * x2 + z1 * r2
    return vcat(r, x, y, z)
end

sdiagm(x, y, z) = SMatrix{3, 3, Float32, 9}(
    x, 0f0, 0f0,
    0f0, y, 0f0,
    0f0, 0f0, z)

gpu_floor(T, x) = unsafe_trunc(T, floor(x))
gpu_ceil(T, x) = unsafe_trunc(T, ceil(x))
gpu_cld(x, y::T) where T = (x + y - one(T)) ÷ y

function get_rect(
    pixel::SVector{2, Float32}, max_radius::Int32,
    grid::SVector{2, Int32}, block::SVector{2, Int32},
)
    rmin = SVector{2, Int32}(
        clamp(gpu_floor(Int32, (pixel[1] - max_radius) / block[1]), 0i32, grid[1]),
        clamp(gpu_floor(Int32, (pixel[2] - max_radius) / block[2]), 0i32, grid[2]))
    rmax = SVector{2, Int32}(
        clamp(gpu_floor(Int32, gpu_cld(pixel[1] + max_radius, block[1])), 0i32, grid[1]),
        clamp(gpu_floor(Int32, gpu_cld(pixel[2] + max_radius, block[2])), 0i32, grid[2]))
    return rmin, rmax
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

ndc2pix(x, S) = ((x + 1f0) * S - 1f0) * 0.5f0

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
    @uniform n = @ndrange()[1]
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

@kernel cpu=false inbounds=true function _permute!(y, @Const(x), @Const(ix))
    i = @index(Global)
    y[i] = x[ix[i]]
end

@kernel cpu=false inbounds=true function duplicate_with_keys!(
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
    radius = radii[i]
    if radius > 0
        rmin, rmax = get_rect(means_2d[i], radius, grid, block)
        # For each tile that the bounding rect overlaps, emit a key/value pair.
        # Key: [tile_id | depth], value: id of the Gaussian.
        # Sorting the values with this key yields Gaussian ids in a list,
        # such that they are first sorted by the tile and then depth.
        depth::UInt64 = reinterpret(UInt32, depths[i])

        offset = i == 1 ? 1i32 : (gaussian_offset[i - 1] + 1i32)
        for y in rmin[2]:(rmax[2] - 1i32), x in rmin[1]:(rmax[1] - 1i32)
            key::UInt64 = UInt64(y) * grid[1] + x
            key <<= 32
            key |= depth
            gaussian_keys[offset] = key
            gaussian_values[offset] = i
            offset += 1
        end
    end
end
