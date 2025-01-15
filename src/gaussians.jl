# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
mutable struct GaussianModel{
    P <: AbstractMatrix{Float32},
    R <: AbstractVector{Int32},
    D <: AbstractArray{Float32, 3},
    G <: AbstractVector{Float32},
    I <: Maybe{AbstractVector{Int32}},
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

    ids::I

    sh_degree::Int
    max_sh_degree::Int
end

function GaussianModel(
    points::AbstractMatrix{Float32}, colors::AbstractMatrix{Float32},
    scales::AbstractMatrix{Float32};
    max_sh_degree::Int = 3, isotropic::Bool = false, use_ids::Bool = false,
)
    0 ≤ max_sh_degree ≤ 3 || throw(ArgumentError(
        "`max_sh_degree=$max_sh_degree` must be in `[0, 3]` range."))

    kab = get_backend(points)
    n = size(points, 2)
    sh_degree = 0

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

    ids = use_ids ? KA.zeros(kab, Int32, n) : nothing

    GaussianModel(
        points, features_dc, features_rest,
        isotropic ? mean(scales; dims=1) : scales,
        rotations, opacities,
        max_radii, accum_∇means_2d, denom, ids,
        sh_degree, max_sh_degree)
end

function GaussianModel(kab; kwargs...)
    points = KA.allocate(kab, Float32, (3, 0))
    scales = KA.allocate(kab, Float32, (3, 0))
    colors = KA.allocate(kab, Float32, (3, 0))
    GaussianModel(points, colors, scales; kwargs...)
end

KernelAbstractions.get_backend(gs::GaussianModel) = get_backend(gs.points)

function bson_params(m::GaussianModel)
    return (;
        points=adapt(CPU(), m.points),
        features_dc=adapt(CPU(), m.features_dc),
        features_rest=adapt(CPU(), m.features_rest),
        scales=adapt(CPU(), m.scales),
        rotations=adapt(CPU(), m.rotations),
        opacities=adapt(CPU(), m.opacities),

        max_radii=adapt(CPU(), m.max_radii),
        accum_∇means_2d=adapt(CPU(), m.accum_∇means_2d),
        denom=adapt(CPU(), m.denom),

        sh_degree=m.sh_degree,
        max_sh_degree=m.max_sh_degree)
end

function set_from_bson!(m::GaussianModel, θ)
    kab = get_backend(m)
    m.points = adapt(kab, θ.points)
    m.features_dc = adapt(kab, θ.features_dc)
    m.features_rest = adapt(kab, θ.features_rest)
    m.scales = adapt(kab, θ.scales)
    m.rotations = adapt(kab, θ.rotations)
    m.opacities = adapt(kab, θ.opacities)

    m.max_radii = adapt(kab, θ.max_radii)
    m.accum_∇means_2d = adapt(kab, θ.accum_∇means_2d)
    m.denom = adapt(kab, θ.denom)

    m.sh_degree = θ.sh_degree
    m.max_sh_degree = θ.max_sh_degree
    return
end

function reset_opacity!(gs::GaussianModel)
    _reset_opacity!(get_backend(gs), 256)(gs.opacities; ndrange=length(gs.opacities))
end

@kernel cpu=false inbounds=true function _reset_opacity!(opacities::AbstractMatrix{Float32})
    i = @index(Global)
    new_opacity = min(0.1f0, NU.sigmoid(opacities[i]))
    opacities[i] = inverse_sigmoid(new_opacity)
end

Base.length(g::GaussianModel) = size(g.points, 2)

function update_stats!(
    gs::GaussianModel, radii::AbstractVector{Int32},
    ∇means_2d::AbstractVector{SVector{2, Float32}},
    resolution::SVector{2, UInt32},
)
    _update_stats!(get_backend(gs), 256)(
        gs.max_radii, gs.accum_∇means_2d, gs.denom,
        radii, ∇means_2d, resolution; ndrange=length(gs))
    return
end

@kernel cpu=false inbounds=true function _update_stats!(
    # Outputs.
    max_radii::AbstractVector{Int32},
    accum_∇means_2d::AbstractVector{Float32},
    denom::AbstractVector{Float32},
    # Inputs.
    radii::AbstractVector{Int32},
    ∇means_2d::AbstractVector{SVector{2, Float32}},
    resolution::SVector{2, UInt32},
)
    i = @index(Global)
    r = radii[i]
    r > 0 || return

    max_radii[i] = max(max_radii[i], r)
    ∇mean_2d = ∇means_2d[i] .* resolution .* 0.5f0
    accum_∇means_2d[i] += norm(∇mean_2d)
    denom[i] += 1f0
end

"""
Convert colors from [0, 1] range to [-SH0 / 2, SH0 / 2].
"""
rgb_2_sh(x) = (x - 0.5f0) * (1f0 / SH0)

sh_2_rgb(x) = x * SH0 + 0.5f0

inverse_sigmoid(x) = log(x / (1f0 - x))

function export_ply!(g::GaussianModel, filename::String)
    ply = PlyIO.Ply()

    n = size(g.points, 2)

    xyz = Array(g.points)
    features_dc = reshape(Array(g.features_dc), :, n)
    features_rest = reshape(Array(g.features_rest), :, n)
    scales = Array(g.scales)
    rotations = Array(g.rotations)
    opacities = reshape(Array(g.opacities), :)

    vertex = PlyIO.PlyElement("vertex",
        PlyIO.ArrayProperty("x", xyz[1, :]),
        PlyIO.ArrayProperty("y", xyz[2, :]),
        PlyIO.ArrayProperty("z", xyz[3, :]),

        [PlyIO.ArrayProperty("f_dc_$i", features_dc[i, :]) for i in 1:size(features_dc, 1)]...,
        [PlyIO.ArrayProperty("f_rest_$i", features_rest[i, :]) for i in 1:size(features_rest, 1)]...,
        [PlyIO.ArrayProperty("scale_$i", scales[i, :]) for i in 1:size(scales, 1)]...,
        [PlyIO.ArrayProperty("rot_$i", rotations[i, :]) for i in 1:size(rotations, 1)]...,

        PlyIO.ArrayProperty("opacity", opacities),
    )
    push!(ply, vertex)

    PlyIO.save_ply(ply, filename)
    return
end

function import_ply(filename::String, kab)
    ply = PlyIO.load_ply(filename)
    vertex = ply["vertex"]

    prop_names = PlyIO.plyname.(vertex.properties)
    n_frest = count(k -> startswith(k, "f_rest_"), prop_names)

    n = length(vertex["x"])
    xyz = vcat([reshape(vertex[i], 1, n) for i in ("x", "y", "z")]...)
    scales = vcat([reshape(vertex["scale_$i"], 1, n) for i in 1:3]...)
    rotations = vcat([reshape(vertex["rot_$i"], 1, n) for i in 1:4]...)
    opacities = reshape(Array(vertex["opacity"]), 1, n)

    features_dc = vcat([reshape(vertex["f_dc_$i"], 1, 1, n) for i in 1:3]...)
    features_rest = if n_frest > 0
        reshape(
            vcat([reshape(vertex["f_rest_$i"], 1, n) for i in 1:n_frest]...),
            3, :, n)
    else
        Array{Float32}(undef, 3, 0, n)
    end

    max_sh_degree::Int = sqrt(size(features_rest, 2) + 1) - 1
    sh_degree::Int = max_sh_degree

    max_radii = KA.zeros(kab, Int32, n)
    accum_∇means_2d = KA.zeros(kab, Float32, n)
    denom = KA.zeros(kab, Float32, n)

    return GaussianModel(
        adapt(kab, xyz), adapt(kab, features_dc), adapt(kab, features_rest),
        adapt(kab, scales), adapt(kab, rotations), adapt(kab, opacities),
        max_radii, accum_∇means_2d, denom,
        nothing, sh_degree, max_sh_degree)
end
