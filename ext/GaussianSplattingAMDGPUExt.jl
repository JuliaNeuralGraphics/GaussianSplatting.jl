module GaussianSplattingAMDGPUExt

using AMDGPU
using GaussianSplatting
using PrecompileTools: @compile_workload

GaussianSplatting.base_array_type(::ROCBackend) = ROCArray

function GaussianSplatting.allocate_pinned(::ROCBackend, ::Type{T}, shape) where T
    x = Array{T}(undef, shape)
    # `own=true` to unregister in dtor what was registered with `hipHostRegister`.
    xd = unsafe_wrap(ROCArray, pointer(x), size(x); own=true)
    return x, xd
end

# Unregistered automatically in the array dtor due to `own=true`.
GaussianSplatting.unpin_memory(::ROCArray) = return

GaussianSplatting.blocking_synchronize(::ROCBackend) = AMDGPU.synchronize(; blocking=true)

@compile_workload begin
    if Base.JLOptions().check_bounds != 1 && Base.JLOptions().code_coverage == 0 &&
        AMDGPU.functional() && !isempty(AMDGPU.devices())
        try
            GaussianSplatting.precompile_workload(ROCBackend())
        catch e
            @debug "AMDGPU precompile workload failed." exception=(e, catch_backtrace())
        end
        GC.gc(true)
    end
end

end
