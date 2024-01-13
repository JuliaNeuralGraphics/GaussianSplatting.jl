# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
struct GeometryState{
    C <: AbstractVector{SVector{6, Float32}},
    D <: AbstractVector{Float32},
    M <: AbstractVector{SVector{2, Float32}},
    R <: AbstractVector{SVector{3, Float32}},
    K <: AbstractVector{SVector{3, Bool}},
    T <: AbstractVector{Int32},
    O <: AbstractVector{SVector{4, Float32}},
    I <: AbstractVector{Int32},
}
    cov3Ds::C
    depths::D
    means_2d::M
    ∇means_2d::M
    rgbs::R
    clamped::K
    tiles_touched::T
    points_offset::T
    conic_opacities::O
    radii::I
end

GeometryState(kab, n::Int) = GeometryState(
    KA.zeros(kab, SVector{6, Float32}, n),
    KA.zeros(kab, Float32, n),
    KA.zeros(kab, SVector{2, Float32}, n),
    KA.zeros(kab, SVector{2, Float32}, n),
    KA.zeros(kab, SVector{3, Float32}, n),
    KA.zeros(kab, SVector{3, Bool}, n),
    KA.zeros(kab, Int32, n),
    KA.zeros(kab, Int32, n),
    KA.zeros(kab, SVector{4, Float32}, n),
    KA.zeros(kab, Int32, n))

Base.length(gstate::GeometryState) = length(gstate.depths)

struct BinningState{
    P <: AbstractVector{UInt32},
    K <: AbstractVector{UInt64},
    V <: AbstractVector{UInt32},
}
    permutation::P
    gaussian_keys_unsorted::K
    gaussian_values_unsorted::V
    gaussian_keys_sorted::K
    gaussian_values_sorted::V
end

BinningState(kab, n::Int) = BinningState(
    KA.zeros(kab, UInt32, n),
    KA.zeros(kab, UInt64, n),
    KA.zeros(kab, UInt32, n),
    KA.zeros(kab, UInt64, n),
    KA.zeros(kab, UInt32, n))

Base.length(gstate::BinningState) = length(gstate.permutation)

struct ImageState{
    R <: AbstractMatrix{UInt32},
    A <: AbstractMatrix{Float32},
}
    ranges::R
    n_contrib::R
    accum_α::A
end

ImageState(kab; width::Int, height::Int, grid_size::Int) = ImageState(
    KA.zeros(kab, UInt32, (2, grid_size)),
    KA.zeros(kab, UInt32, (width, height)),
    KA.zeros(kab, Float32, (width, height)))
