# Checkpoints are safetensors files: a flat `name -> tensor` table with a JSON header.
#
# The format is flat & string-keyed, so the structure lives in the names:
# every array is a tensor under a dotted prefix
# (`gaussians.points`, `optimizers.scales.mu.1`, ...) &
# every scalar is a string in the `__metadata__` map.
# Optional groups (the sky dome, the bilateral grid) are simply absent, which `haskey` detects.
#
# Each type contributes a pair of methods:
#
#     write_state!(tensors, meta, prefix, x)   # save
#     read_state!(x, ckpt, prefix)             # load, in place
#
# See `save_state` / `load_state!` in `training.jl` for the top-level document.
const CHECKPOINT_FORMAT = "GaussianSplatting.jl-checkpoint-1"

"""
A checkpoint opened for reading:
`st` holds the lazy (mmap-backed) tensors, `meta` the scalars.
Read from it with [`tensor`](@ref) & the `read_*` helpers.
"""
struct Checkpoint
    st::SafeTensors.SafeTensor
    meta::Dict{String, String}
end

Base.haskey(ckpt::Checkpoint, key::String) = haskey(ckpt.st, key)

"""
Materialize the tensor at `key` as a host array.
Tensors are stored in C order, so this also un-permutes back to the column-major array that was written.
"""
tensor(ckpt::Checkpoint, key::String) = collect(ckpt.st[key])::Array

# Scalars go through their `string` representation, which round-trips exactly
# for `Float32` (shortest form that parses back to the same value).
write_scalar!(meta::Dict{String, String}, key::String, x) = meta[key] = string(x)
write_vec!(meta::Dict{String, String}, key::String, v) = meta[key] = join(v, ",")
read_scalar(ckpt::Checkpoint, key::String, ::Type{T}) where T <: Number = parse(T, ckpt.meta[key])
read_vec(ckpt::Checkpoint, key::String, ::Type{SVector{N, T}}) where {N, T} = SVector{N, T}(parse.(T, split(ckpt.meta[key], ","))...)

"""
    save_checkpoint(filename, tensors, meta)

Write `tensors` (arrays, by dotted name) & `meta` (scalars, as strings) to `filename`.
Read back with [`load_checkpoint`](@ref).
"""
function save_checkpoint(
    filename::String, tensors::AbstractDict{String, <:AbstractArray},
    meta::Dict{String, String},
)
    meta["format"] = CHECKPOINT_FORMAT
    SafeTensors.serialize(filename, tensors, meta)
    return
end

"""
Counterpart of [`save_checkpoint`](@ref).

The returned [`Checkpoint`](@ref) borrows the file's memory mapping:
reading a tensor out of it touches the disk, so keep it alive until done with it.
"""
function load_checkpoint(filename::String)
    st = SafeTensors.deserialize(filename)
    meta = st.metadata
    (meta ≢ nothing && get(meta, "format", nothing) == CHECKPOINT_FORMAT) || throw(ArgumentError(
        "`$filename` is not a GaussianSplatting.jl checkpoint " *
        "(no `$CHECKPOINT_FORMAT` in its metadata)."))
    return Checkpoint(st, meta)
end
