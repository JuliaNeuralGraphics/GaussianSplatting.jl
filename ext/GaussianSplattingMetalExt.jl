module GaussianSplattingMetalExt

using Metal
using GaussianSplatting
using PrecompileTools: @compile_workload

GaussianSplatting.base_array_type(::MetalBackend) = MtlArray

function GaussianSplatting.allocate_pinned(::MetalBackend, ::Type{T}, shape) where T
    xd = MtlArray{T, length(shape), Metal.SharedStorage}(undef, shape)
    x = reshape(unsafe_wrap(Vector{T}, reshape(xd, :)), shape)
    return x, xd
end

# Unregistered automatically in the array dtor.
GaussianSplatting.unpin_memory(::MtlArray) = return

@compile_workload begin
    if Base.JLOptions().check_bounds != 1 && Base.JLOptions().code_coverage == 0 &&
        Metal.functional() && !isempty(Metal.devices())
        try
            GaussianSplatting.precompile_workload(MetalBackend())
        catch e
            @debug "Metal precompile workload failed." exception=(e, catch_backtrace())
        end
        GC.gc(true)
    end
end

end
