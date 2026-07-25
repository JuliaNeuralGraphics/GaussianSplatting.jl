module GaussianSplattingAMDGPUExt

using AMDGPU
using GaussianSplatting

GaussianSplatting.base_array_type(::ROCBackend) = ROCArray

GaussianSplatting.use_ak(::ROCBackend) = true

function GaussianSplatting.allocate_pinned(::ROCBackend, ::Type{T}, shape) where T
    x = Array{T}(undef, shape)
    # `own=true` is what makes the dtor release the `hipHostRegister` mapping
    # that `unsafe_wrap` takes out (it unregisters, it does not free Julia's
    # memory). Without it the mapping outlives the `Array`, and once GC recycles
    # that address a later host allocation lands on a stale registration and
    # copies off it fail with `hipErrorInvalidValue`.
    xd = unsafe_wrap(ROCArray, pointer(x), size(x); own=true)
    return x, xd
end

# Unregistered automatically in the array dtor due to `own=true`.
GaussianSplatting.unpin_memory(::ROCArray) = return

GaussianSplatting.blocking_synchronize(::ROCBackend) = AMDGPU.synchronize(; blocking=true)

end
