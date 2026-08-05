# TOML (de)serialization for `OptimizationParams`.

const OPT_PARAM_ENUMS = (
    depth_loss_mode = (:ssi, :ssi_disparity, :ssi_depth),
    sky_dome_shape = SKY_DOME_SHAPES,
)

"""
    save_opt_params(path::AbstractString, params::OptimizationParams)

Write `params` as TOML: every field, sorted, so two runs diff cleanly.
"""
function save_opt_params(path::AbstractString, params::OptimizationParams)
    entries = Dict{String, Any}(
        String(name) => to_toml(getfield(params, name))
        for name in fieldnames(OptimizationParams))

    open(path, "w") do io
        println(io, "# GaussianSplatting.jl `OptimizationParams`.")
        println(io, "# Load with `Open Dataset` in the GUI, or `load_opt_params`.")
        TOML.print(io, entries; sorted=true)
    end
    return path
end

# TOML has no symbols & no tuples.
to_toml(x::Symbol) = String(x)
to_toml(x::Tuple) = collect(x)
to_toml(x::Float32) = parse(Float64, string(x))
to_toml(x) = x

"""
    with_params(params::OptimizationParams; changes...)

Copy of `params` with `changes` applied.
"""
with_params(params::OptimizationParams; changes...) = OptimizationParams(;
    (name => get(changes, name, getfield(params, name))
    for name in fieldnames(OptimizationParams))...)

"""
    load_opt_params(path::AbstractString)::OptimizationParams

Read an `OptimizationParams` from a TOML file.
Omitted fields keep their default, so a file may specify only what a run changes.

Throws on an unknown key or a value of the wrong type.
"""
function load_opt_params(path::AbstractString)
    entries = TOML.parsefile(path)

    known = fieldnames(OptimizationParams)
    kwargs = Pair{Symbol, Any}[]
    for (key, value) in entries
        name = Symbol(key)
        name in known || error("Unknown `OptimizationParams` field: `$key`. Valid fields: $(join(known, ", ")).")
        T = fieldtype(OptimizationParams, name)
        push!(kwargs, name => from_toml(T, name, value))
    end
    return OptimizationParams(; kwargs...)
end

# TOML integers parse as `Int64` & floats as `Float64` - narrow them to the field's type.
from_toml(::Type{Float32}, name::Symbol, value::Real) = Float32(value)
from_toml(::Type{Int}, name::Symbol, value::Integer) = Int(value)
from_toml(::Type{Bool}, name::Symbol, value::Bool) = value

function from_toml(::Type{Symbol}, name::Symbol, value::AbstractString)
    sym = Symbol(value)
    allowed = get(OPT_PARAM_ENUMS, name, nothing)
    (allowed ≡ nothing || sym in allowed) || error("Invalid `$name`: `$value` ∉ $allowed.")
    return sym
end

function from_toml(::Type{NTuple{N, Int}}, name::Symbol, value::AbstractVector) where N
    length(value) == N || error("`$name` needs exactly $N integers, got $(length(value)).")
    all(v -> v isa Integer, value) || error("`$name` must be integers, got `$value`.")
    return NTuple{N, Int}(value)
end

from_toml(::Type{T}, name::Symbol, value) where T =
    error("`$name` must be a `$T`, got `$(repr(value))`::$(typeof(value)).")
