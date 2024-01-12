mutable struct GaussianModel{
    P <: AbstractMatrix{Float32},
    R <: AbstractVector{Int32},
    D <: AbstractArray{Float32, 3},
    G <: AbstractVector{Float32},
}
    points::P
    features_dc::D
    features_rest::D
    scales::P
    rotations::P
    opacities::P

    max_radii::R
    accum_∇means_2d::G
    denom::G

    sh_degree::Int
    max_sh_degree::Int
end

function GaussianModel(
    points::AbstractMatrix{Float32}, colors::AbstractMatrix{Float32},
    scales::AbstractMatrix{Float32},
)
    kab = get_backend(points)
    n = size(points, 2)
    sh_degree, max_sh_degree = 0, 3

    colors = rgb_2_sh.(colors)
    n_features = (max_sh_degree + 1)^2
    features = KA.zeros(kab, Float32, (3, n_features, n))
    features[:, 1, :] .= colors
    features_dc = features[:, [1], :]
    features_rest = features[:, 2:end, :]

    # Intial rotation is an identity rotation: (1, 0, 0, 0) quaternion.
    rotations = KA.zeros(kab, Float32, (4, n))
    rotations[1, :] .= 1f0

    opacities = inverse_sigmoid.(0.1f0 .* KA.ones(kab, Float32, (1, n)))
    max_radii = KA.zeros(kab, Int32, n)
    accum_∇means_2d = KA.zeros(kab, Float32, n)
    denom = KA.zeros(kab, Float32, n)
    GaussianModel(
        points, features_dc, features_rest,
        scales, rotations, opacities, max_radii, accum_∇means_2d, denom,
        sh_degree, max_sh_degree)
end

scales_activation = exp
scales_inv_activation = log
opacity_activation = NU.sigmoid

KernelAbstractions.get_backend(gs::GaussianModel) = get_backend(gs.points)

function reset_opacity!(gs::GaussianModel)
    _reset_opacity!(get_backend(gs), 256)(gs.opacities; ndrange=length(gs.opacities))
end

@kernel function _reset_opacity!(opacities::AbstractMatrix{Float32})
    i = @index(Global)
    @inbounds new_opacity = min(0.1f0, NU.sigmoid(opacities[i]))
    @inbounds opacities[i] = inverse_sigmoid(new_opacity)
end

function update_stats!(
    gs::GaussianModel, radii::AbstractVector{Int32},
    ∇means_2d::AbstractVector{SVector{2, Float32}},
)
    _update_stats!(get_backend(gs), 256)(
        gs.max_radii, gs.accum_∇means_2d, gs.denom,
        radii, ∇means_2d; ndrange=length(radii))
    return
end

@kernel function _update_stats!(
    # Outputs.
    max_radii::AbstractVector{Int32},
    accum_∇means_2d::AbstractVector{Float32},
    denom::AbstractVector{Float32},
    # Inputs.
    radii::AbstractVector{Int32},
    ∇means_2d::AbstractVector{SVector{2, Float32}},
)
    i = @index(Global)
    @inbounds r = radii[i]
    if r > 0
        @inbounds max_radii[i] = max(max_radii[i], r)
        @inbounds accum_∇means_2d[i] += norm(∇means_2d[i])
        @inbounds denom[i] += 1f0
    end
end

"""
Convert colors from [0, 1] range to [-SH0 / 2, SH0 / 2].
"""
rgb_2_sh(x) = (x - 0.5f0) * (1f0 / SH0)

sh_2_rgb(x) = x * SH0 + 0.5f0

inverse_sigmoid(x) = log(x / (1f0 - x))
