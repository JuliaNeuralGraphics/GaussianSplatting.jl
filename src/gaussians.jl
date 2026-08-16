# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
mutable struct GaussianModel{
    P <: AbstractMatrix{Float32},
    D <: AbstractArray{Float32, 3},
    I <: Maybe{AbstractVector{Int32}},
}
    points::P
    features_dc::D
    features_rest::D
    scales::P
    rotations::P
    opacities::P

    ids::I

    sh_degree::Int
    max_sh_degree::Int
end

# Initialize a model on `kab` from a point cloud, which may live on any
# backend (a dataset's cloud is a host array).
function GaussianModel(kab,
    points::AbstractMatrix{Float32}, colors::AbstractMatrix{Float32},
    scales::AbstractMatrix{Float32};
    max_sh_degree::Int = 3, isotropic::Bool = false, use_ids::Bool = false,
)
    0 ≤ max_sh_degree ≤ 3 || throw(ArgumentError(
        "`max_sh_degree=$max_sh_degree` must be in `[0, 3]` range."))

    n = size(points, 2)
    sh_degree = 0

    colors = adapt(kab, rgb_2_sh.(colors))
    n_features = (max_sh_degree + 1)^2
    features = KA.zeros(kab, Float32, (3, n_features, n))
    features[:, 1, :] .= colors
    features_dc = features[:, [1], :]
    features_rest = features[:, 2:end, :]

    # Intial rotation is an identity rotation: (1, 0, 0, 0) quaternion.
    rotations = KA.zeros(kab, Float32, (4, n))
    rotations[1, :] .= 1f0

    opacities = inverse_sigmoid.(0.1f0 .* KA.ones(kab, Float32, (1, n)))

    ids = use_ids ? KA.zeros(kab, Int32, n) : nothing

    # The model owns its arrays (densification replaces & frees them),
    # so it must not alias the caller's `points`/`scales` (e.g. a dataset's
    # point cloud, which outlives the model).
    GaussianModel(
        adopt(kab, points), features_dc, features_rest,
        isotropic ? mean(adapt(kab, scales); dims=1) : adopt(kab, scales),
        rotations, opacities, ids,
        sh_degree, max_sh_degree)
end

function GaussianModel(kab; kwargs...)
    empty = Matrix{Float32}(undef, 3, 0)
    GaussianModel(kab, empty, empty, empty; kwargs...)
end

"""
Move `x` onto `kab`, always returning an array the caller does not share:
`adapt` is the identity when `x` already lives on `kab`, which would alias it.
"""
adopt(kab, x::AbstractArray) = (y = adapt(kab, x); y ≡ x ? copy(x) : y)

KernelAbstractions.get_backend(gs::GaussianModel) = get_backend(gs.points)

memory_usage(gs::GaussianModel) =
    memory_usage(gs.points) +
    memory_usage(gs.features_dc) +
    memory_usage(gs.features_rest) +
    memory_usage(gs.scales) +
    memory_usage(gs.rotations) +
    memory_usage(gs.opacities) +
    memory_usage(gs.ids)

function KA.unsafe_free!(gs::GaussianModel)
    KA.unsafe_free!(gs.points)
    KA.unsafe_free!(gs.features_dc)
    KA.unsafe_free!(gs.features_rest)
    KA.unsafe_free!(gs.scales)
    KA.unsafe_free!(gs.rotations)
    KA.unsafe_free!(gs.opacities)
    isnothing(gs.ids) || KA.unsafe_free!(gs.ids)
    return
end

function write_state!(tensors, meta, prefix::String, m::GaussianModel)
    tensors["$prefix.points"] = adapt(CPU(), m.points)
    tensors["$prefix.features_dc"] = adapt(CPU(), m.features_dc)
    tensors["$prefix.features_rest"] = adapt(CPU(), m.features_rest)
    tensors["$prefix.scales"] = adapt(CPU(), m.scales)
    tensors["$prefix.rotations"] = adapt(CPU(), m.rotations)
    tensors["$prefix.opacities"] = adapt(CPU(), m.opacities)

    write_scalar!(meta, "$prefix.sh_degree", m.sh_degree)
    write_scalar!(meta, "$prefix.max_sh_degree", m.max_sh_degree)
    return
end

function read_state!(m::GaussianModel, ckpt::Checkpoint, prefix::String)
    kab = get_backend(m)
    m.points = adapt(kab, tensor(ckpt, "$prefix.points"))
    m.features_dc = adapt(kab, tensor(ckpt, "$prefix.features_dc"))
    m.features_rest = adapt(kab, tensor(ckpt, "$prefix.features_rest"))
    m.scales = adapt(kab, tensor(ckpt, "$prefix.scales"))
    m.rotations = adapt(kab, tensor(ckpt, "$prefix.rotations"))
    m.opacities = adapt(kab, tensor(ckpt, "$prefix.opacities"))

    m.sh_degree = read_scalar(ckpt, "$prefix.sh_degree", Int)
    m.max_sh_degree = read_scalar(ckpt, "$prefix.max_sh_degree", Int)
    return
end

"""
Cap every opacity at `value`, leaving the already-fainter ones alone.
The periodic reset that lets 3DGS-style pruning cull Gaussians which never
earn their opacity back.
"""
function reset_opacity!(gs::GaussianModel, value::Float32 = 0.1f0)
    _reset_opacity!(get_backend(gs), 256)(gs.opacities, value; ndrange=length(gs.opacities))
end

@kernel cpu=false inbounds=true function _reset_opacity!(
    opacities::AbstractMatrix{Float32}, value::Float32,
)
    i = @index(Global)
    new_opacity = min(value, NU.sigmoid(opacities[i]))
    opacities[i] = inverse_sigmoid(new_opacity)
end

Base.length(g::GaussianModel) = size(g.points, 2)

"""
Convert colors from [0, 1] range to [-SH0 / 2, SH0 / 2].
"""
rgb_2_sh(x) = (x - 0.5f0) * (1f0 / SH0)

sh_2_rgb(x) = x * SH0 + 0.5f0

inverse_sigmoid(x) = log(x / (1f0 - x))

"""
Write `g` as a binary `.ply` in the layout the reference 3DGS implementation
writes, which is what external viewers & editors read:

    x y z  nx ny nz  f_dc_0..2  f_rest_0..N  opacity  scale_0..2  rot_0..3

Two details are load-bearing for those readers & are the reason this does not
go through `PlyIO.save_ply`:

- The type is spelled `float`. PlyIO emits `float32`, which is not a canonical
  PLY type name & is rejected outright by some readers.
- `f_rest` is channel-major: all coefficients of R, then of G, then of B,
  matching the reference's `transpose(1, 2).flatten()`. The model stores the
  transpose of that.

Normals are written as zeros, as the reference does: nothing consumes them,
but readers key off the property set.
"""
function export_ply(g::GaussianModel, filename::String)
    n = size(g.points, 2)

    xyz = Array(g.points)
    features_dc = reshape(Array(g.features_dc), :, n)
    # (channel, coefficient, gaussian) -> (coefficient, channel, gaussian),
    # so that flattening gives the channel-major order described above.
    features_rest = reshape(
        permutedims(Array(g.features_rest), (2, 1, 3)), :, n)
    scales = Array(g.scales)
    rotations = Array(g.rotations)
    opacities = reshape(Array(g.opacities), 1, n)

    names = vcat(
        ["x", "y", "z"],
        ["nx", "ny", "nz"],
        ["f_dc_$(i - 1)" for i in axes(features_dc, 1)],
        ["f_rest_$(i - 1)" for i in axes(features_rest, 1)],
        ["opacity"],
        ["scale_$(i - 1)" for i in axes(scales, 1)],
        ["rot_$(i - 1)" for i in axes(rotations, 1)])
    # One row per property, in `names` order. Julia is column-major, so this
    # is already the interleaved layout PLY stores: every property of the
    # first vertex, then of the second, ...
    properties = vcat(
        xyz,
        zeros(Float32, 3, n), # Normals.
        features_dc, features_rest, opacities, scales, rotations)
    @assert size(properties, 1) == length(names)

    format = ENDIAN_BOM == 0x04030201 ?
        "binary_little_endian" : "binary_big_endian"
    open(filename, "w") do io
        println(io, "ply")
        println(io, "format $format 1.0")
        println(io, "element vertex $n")
        for name in names
            println(io, "property float $name")
        end
        println(io, "end_header")
        write(io, properties)
    end
    return
end

"""
Read a `.ply` written by [`export_ply`](@ref) or by any other 3DGS
implementation, onto `kab`. See [`export_ply`](@ref) for the layout; both
the property order in the header & the storage precision are free, only the
names matter here.
"""
function import_ply(filename::String, kab)
    ply = PlyIO.load_ply(filename)
    vertex = ply["vertex"]

    prop_names = PlyIO.plyname.(vertex.properties)
    n_frest = count(k -> startswith(k, "f_rest_"), prop_names)
    n_frest % 3 == 0 || throw(ArgumentError(
        "`$filename` has $n_frest `f_rest_*` properties, which is not a " *
        "whole number of SH coefficients per color channel."))

    n = length(vertex["x"])
    # A row of one property, converted from whatever the file stores it as.
    row(name) = reshape(Float32.(vertex[name]), 1, n)

    xyz = vcat((row(i) for i in ("x", "y", "z"))...)
    scales = vcat((row("scale_$(i - 1)") for i in 1:3)...)
    rotations = vcat((row("rot_$(i - 1)") for i in 1:4)...)
    opacities = row("opacity")

    features_dc = reshape(vcat((row("f_dc_$(i - 1)") for i in 1:3)...), 3, 1, n)
    features_rest = if n_frest > 0
        # Channel-major in the file (see `export_ply`), (channel,
        # coefficient, gaussian) in the model.
        permutedims(
            reshape(vcat((row("f_rest_$(i - 1)") for i in 1:n_frest)...), :, 3, n),
            (2, 1, 3))
    else
        Array{Float32}(undef, 3, 0, n)
    end

    max_sh_degree::Int = sqrt(size(features_rest, 2) + 1) - 1
    sh_degree::Int = max_sh_degree

    gaussians = GaussianModel(
        adapt(kab, xyz), adapt(kab, features_dc), adapt(kab, features_rest),
        adapt(kab, scales), adapt(kab, rotations), adapt(kab, opacities),
        nothing, sh_degree, max_sh_degree)

    return (; gaussians, vertex)
end
