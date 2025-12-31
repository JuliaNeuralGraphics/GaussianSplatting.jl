module GaussianSplattingMetalExt

using Metal
using GaussianSplatting

GaussianSplatting.base_array_type(::MetalBackend) = MtlArray

GaussianSplatting.use_ak(::MetalBackend) = true

function GaussianSplatting.allocate_pinned(::MetalBackend, ::Type{T}, shape) where T
    xd = MtlArray{T, length(shape), Metal.SharedStorage}(undef, shape)
    x = reshape(unsafe_wrap(Vector{T}, reshape(xd, :)), shape)
    return x, xd
end

# Unregistered automatically in the array dtor.
GaussianSplatting.unpin_memory(::MtlArray) = return

end
