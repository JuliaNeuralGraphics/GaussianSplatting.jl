# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
struct GeometryState{
    D <: AbstractVector{Float32},
    M <: AbstractVector{SVector{2, Float32}},
    R <: AbstractVector{SVector{3, Float32}},
    K <: AbstractVector{SVector{3, Bool}},
    T <: AbstractVector{Int32},
    O <: AbstractVector{SVector{3, Float32}},
    I <: AbstractVector{Int32},
    F <: Maybe{AbstractMatrix{Float32}},
    N <: Maybe{AbstractVector{SVector{3, Float32}}},
}
    depths::D
    means_2d::M
    ∇means_2d::M
    rgbs::R
    clamped::K
    tiles_touched::T
    points_offset::T
    conic_opacities::O
    radii::I
    # Blended feature channels, `n_color_features(mode)` rows:
    #   rgb (1:3), depth (4), constant-1 alpha feature (5), camera-space
    #   normal (6:8, `:rgbdn` only).
    # The constant-1 channel blends to `Σᵢ 1·αᵢ·Tᵢ = 1 - T_final`, so it
    # renders the alpha map & the standard per-channel backward yields its
    # exact gradient (the `grad_alpha` path).
    color_features::F
    # Staging buffer for the per-Gaussian normals written by `project!`,
    # copied into rows 6:8 of `color_features` (`:rgbdn` only).
    normals::N
end

function GeometryState(kab, n::Int; n_features::Int = 0)
    GeometryState(
        KA.zeros(kab, Float32, n),
        KA.zeros(kab, SVector{2, Float32}, n),
        KA.zeros(kab, SVector{2, Float32}, n),
        KA.zeros(kab, SVector{3, Float32}, n),
        KA.zeros(kab, SVector{3, Bool}, n),
        KA.zeros(kab, Int32, n),
        KA.zeros(kab, Int32, n),
        KA.zeros(kab, SVector{3, Float32}, n),
        KA.zeros(kab, Int32, n),
        n_features > 3 ? KA.zeros(kab, Float32, (n_features, n)) : nothing,
        n_features > 5 ? KA.zeros(kab, SVector{3, Float32}, n) : nothing)
end

Base.length(gstate::GeometryState) = length(gstate.depths)

function KA.unsafe_free!(gstate::GeometryState)
    KA.unsafe_free!(gstate.depths)
    KA.unsafe_free!(gstate.means_2d)
    KA.unsafe_free!(gstate.∇means_2d)
    KA.unsafe_free!(gstate.rgbs)
    KA.unsafe_free!(gstate.clamped)
    KA.unsafe_free!(gstate.tiles_touched)
    KA.unsafe_free!(gstate.points_offset)
    KA.unsafe_free!(gstate.conic_opacities)
    KA.unsafe_free!(gstate.radii)
    isnothing(gstate.color_features) || KA.unsafe_free!(gstate.color_features)
    isnothing(gstate.normals) || KA.unsafe_free!(gstate.normals)
    return
end

struct BinningState{
    P <: AbstractVector{UInt32},
    K <: AbstractVector{UInt64},
    V <: AbstractVector{UInt32},
}
    permutation::P
    permutation_tmp::P
    gaussian_keys_unsorted::K
    gaussian_values_unsorted::V
    gaussian_keys_sorted::K
    gaussian_values_sorted::V
end

BinningState(kab, n::Int) = BinningState(
    KA.zeros(kab, UInt32, n),
    KA.zeros(kab, UInt32, n),
    KA.zeros(kab, UInt64, n),
    KA.zeros(kab, UInt32, n),
    KA.zeros(kab, UInt64, n),
    KA.zeros(kab, UInt32, n))

Base.length(gstate::BinningState) = length(gstate.permutation)

function KA.unsafe_free!(bstate::BinningState)
    KA.unsafe_free!(bstate.permutation)
    KA.unsafe_free!(bstate.permutation_tmp)
    KA.unsafe_free!(bstate.gaussian_keys_unsorted)
    KA.unsafe_free!(bstate.gaussian_values_unsorted)
    KA.unsafe_free!(bstate.gaussian_keys_sorted)
    KA.unsafe_free!(bstate.gaussian_values_sorted)
    return
end

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

# All fields are arrays (or `nothing`), so this stays correct as they change.
memory_usage(s::Union{GeometryState, BinningState, ImageState}) =
    sum(f -> memory_usage(getfield(s, f)), fieldnames(typeof(s)); init=0)

function KA.unsafe_free!(istate::ImageState)
    KA.unsafe_free!(istate.ranges)
    KA.unsafe_free!(istate.n_contrib)
    KA.unsafe_free!(istate.accum_α)
    return
end
