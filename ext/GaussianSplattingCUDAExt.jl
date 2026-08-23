module GaussianSplattingCUDAExt

using CUDA
using GaussianSplatting
using PrecompileTools: @compile_workload

GaussianSplatting.base_array_type(::CUDABackend) = CuArray

function GaussianSplatting.allocate_pinned(::CUDABackend, ::Type{T}, shape) where T
    x = Array{T}(undef, shape)
    buf = CUDA.register(CUDA.HostMemory, pointer(x), sizeof(x), CUDA.MEMHOSTREGISTER_DEVICEMAP)
    xptr = convert(CuPtr{Float32}, buf)
    xd = unsafe_wrap(CuArray, xptr, size(x))
    return x, xd
end

function GaussianSplatting.unpin_memory(x::CuArray)
    CUDA.unregister(x.data.rc.obj.mem)
    return
end

@compile_workload begin
    if Base.JLOptions().check_bounds != 1 && Base.JLOptions().code_coverage == 0 &&
        CUDA.functional() && !isempty(CUDA.devices())
        try
            GaussianSplatting.precompile_workload(CUDABackend())
        catch e
            @debug "CUDA precompile workload failed." exception=(e, catch_backtrace())
        end
        GC.gc(true)
    end
end

end
