# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
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
    radius > 0 || return

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

@kernel cpu=false inbounds=true function count_tiles_per_gaussian!(
    # Output.
    tiles_touched::AbstractVector{Int32},
    # Input.
    means_2D::AbstractVector{SVector{2, Float32}},
    radii::AbstractVector{Int32},
    tile_grid::SVector{2, Int32},
    tile_size::SVector{2, Int32},
)
    i = @index(Global)
    radius = radii[i]
    if !(radius > 0f0)
        tiles_touched[i] = 0i32
        return
    end

    mean_2D = means_2D[i]
    rect_min, rect_max = get_rect(mean_2D, radius, tile_grid, tile_size)
    area = prod(rect_max .- rect_min)
    tiles_touched[i] = area
end
