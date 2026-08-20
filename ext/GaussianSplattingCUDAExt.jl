module GaussianSplattingCUDAExt

using CUDA
using GaussianSplatting

GaussianSplatting.base_array_type(::CUDABackend) = CuArray

function GaussianSplatting.allocate_pinned(::CUDABackend, ::Type{T}, shape) where T
    x = Array{T}(undef, shape)
    buf = CUDA.register(CUDA.HostMemory, pointer(x), sizeof(x),
        CUDA.MEMHOSTREGISTER_DEVICEMAP)
    xptr = convert(CuPtr{Float32}, buf)
    xd = unsafe_wrap(CuArray, xptr, size(x))
    return x, xd
end

function GaussianSplatting.unpin_memory(x::CuArray)
    CUDA.unregister(x.data.rc.obj.mem)
    return
end

end
