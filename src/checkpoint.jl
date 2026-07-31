# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

# BSON writes a document's total length into an `Int32` field, so a document
# whose contents sum past 2 GiB cannot be written at all - it fails with an
# `InexactError` on the length. A few million Gaussians plus their Adam moments
# reach that (~700 B per Gaussian), so the bulk arrays go into a raw payload
# region & the BSON document keeps only the structure, with a `BlobRef` in
# place of each array. BSON is good at the structure (nested named tuples,
# `Camera`, ...) & indifferent to the arrays, so this plays to its strengths.
#
# It also sidesteps `bson_doc` buffering the whole document in memory (and then
# copying it) before it reaches the file: blobs stream straight out.
#
# Layout, all in one file:
#
#     "GSPCKPT1"      magic, 8 bytes
#     header_offset   `UInt64`, patched in once the payload size is known
#     payload         every array, back to back
#     header          BSON document, arrays replaced by `BlobRef`s
#
# Files that do not start with the magic are read as plain BSON, so
# checkpoints written before this format still load.
const CHECKPOINT_MAGIC = b"GSPCKPT1"

"""
Stands in for an array in the BSON header. `offset` is an absolute file
position, so restoring an array is a seek & a read.
"""
struct BlobRef
    offset::Int64
    dims::Vector{Int}
    eltype::DataType
end

# Arrays are written to `io` & replaced by a `BlobRef`; everything else is left
# to BSON. The numeric-array method is the more specific one, so a
# `Vector{Float32}` becomes a blob while a `Vector{<:Array}` is descended into.
externalize!(io::IO, x) = x
externalize!(io::IO, x::AbstractDict) =
    Dict(k => externalize!(io, v) for (k, v) in x)
externalize!(io::IO, x::NamedTuple) = map(v -> externalize!(io, v), x)
externalize!(io::IO, x::Vector) = [externalize!(io, v) for v in x]

function externalize!(io::IO, x::Array{T}) where T <: Number
    ref = BlobRef(position(io), collect(size(x)), T)
    write(io, x)
    return ref
end

internalize(io::IO, x) = x
internalize(io::IO, x::AbstractDict) =
    Dict(k => internalize(io, v) for (k, v) in x)
internalize(io::IO, x::NamedTuple) = map(v -> internalize(io, v), x)
internalize(io::IO, x::Vector) = [internalize(io, v) for v in x]

function internalize(io::IO, r::BlobRef)
    seek(io, r.offset)
    return read!(io, Array{r.eltype}(undef, r.dims...))
end

"""
Write `doc` to `filename`, keeping its arrays out of the BSON document so the
2 GiB document limit does not apply. Read back with [`load_checkpoint`](@ref).
"""
function save_checkpoint(filename::String, doc::AbstractDict)
    open(filename, "w") do io
        write(io, CHECKPOINT_MAGIC)
        # The header lands after the payload, whose size is unknown until it
        # has been written: leave room & patch the offset in at the end.
        offset_pos = position(io)
        write(io, UInt64(0))

        header = externalize!(io, doc)
        header_offset = position(io)
        BSON.bson(io, header)

        seek(io, offset_pos)
        write(io, UInt64(header_offset))
    end
    return
end

"""
Counterpart of [`save_checkpoint`](@ref). Also reads plain BSON files, which is
what checkpoints written before the blob format are.
"""
function load_checkpoint(filename::String)
    open(filename, "r") do io
        read(io, length(CHECKPOINT_MAGIC)) == CHECKPOINT_MAGIC ||
            return BSON.load(filename)

        seek(io, read(io, UInt64))
        return internalize(io, BSON.load(io))
    end
end
