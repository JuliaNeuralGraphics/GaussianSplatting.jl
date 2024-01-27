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