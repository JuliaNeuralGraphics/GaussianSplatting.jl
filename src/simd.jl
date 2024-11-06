@inline function vload(::Type{SIMD.Vec{N, T}}, ptr::Core.LLVMPtr{T, AS}) where {N, T, AS}
    alignment = sizeof(T) * N
    vec_ptr = Base.bitcast(Core.LLVMPtr{NTuple{N, T}, AS}, ptr)
    return unsafe_load(vec_ptr, 1, Val(alignment))
end

@inline function vload(ptr::Core.LLVMPtr{SVector{N, T}, AS}) where {N, T, AS}
    raw_ptr = reinterpret(Core.LLVMPtr{T, AS}, ptr)
    val = vload(SIMD.Vec{N, T}, raw_ptr)
    return SVector{N, T}(val)
end

@inline function vstore!(ptr::Core.LLVMPtr{T, AS}, x::NTuple{N, T}) where {N, T, AS}
    alignment = sizeof(T) * N
    vec_ptr = Base.bitcast(Core.LLVMPtr{NTuple{N, T}, AS}, ptr)
    unsafe_store!(vec_ptr, x, 1, Val(alignment))
    return
end

@inline function vstore!(ptr::Core.LLVMPtr{SVector{N, T}, AS}, x::SVector{N, T}) where {N, T, AS}
    raw_ptr = reinterpret(Core.LLVMPtr{T, AS}, ptr)
    val = tuple(x...) # TODO avoid
    vstore!(raw_ptr, val)
end
