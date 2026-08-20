"""
    sparse_step!(opt::NU.Adam, θ, ∇, radii::AbstractVector{Int32}; dispose::Bool)

Like `NU.step!`, but skips the entire update (no momentum decay, no
parameter write) for gaussians with `radii[g] == 0` — i.e. gaussians not
visible/touched by the current view's rasterization pass. `θ`/`∇` must be
one of the per-gaussian parameter tensors, whose last dimension is the
gaussian index.

`radii` may be *longer* than that: the rasterizer's `GeometryState` is
grow-only, so after a strategy prunes gaussians it still carries the
entries of the pre-prune count. Only its first `size(θ)[end]` entries are
read. The stride is derived from `θ` rather than from `length(radii)`,
since the latter silently yields a wrong stride whenever it happens to
divide `length(θ)`.
"""
function sparse_step!(
    opt::NU.Adam, θ::T, ∇::T, radii::AbstractVector{Int32}; dispose::Bool,
) where T <: AbstractArray
    size(θ) == size(∇) || throw(ArgumentError(
        "Shape of parameters and gradients must be the same, " *
        "instead: `$(size(θ))` vs `$(size(∇))`."))

    n_gaussians = size(θ, ndims(θ))
    n_gaussians ≤ length(radii) || throw(ArgumentError(
        "`radii` is shorter than the gaussian dimension of `θ`: " *
        "`length(radii) = $(length(radii))` vs `size(θ)[end] = $n_gaussians`."))
    stride = length(θ) ÷ n_gaussians

    # Bump `current_step`, but don't use it for bias correction.
    # Otherwise it requires per-gaussian counter to produce correct updates.
    # In practice, bias correction shouldn't matter much here.
    opt.current_step += 0x1
    sparse_adam_step_kernel!(KA.get_backend(opt))(
        opt.μ[1], opt.ν[1], θ, ∇, radii, stride,
        opt.lr, opt.β1, opt.β2, opt.ϵ; ndrange=length(θ))

    dispose && KA.unsafe_free!(∇)
    return
end

@kernel cpu=false inbounds=true function sparse_adam_step_kernel!(
    μ, ν, Θ, @Const(∇), @Const(radii), stride::Int, lr::Float32,
    β1::Float32, β2::Float32, ϵ::Float32,
)
    i = @index(Global)
    g = (i - 1) ÷ stride + 1
    radii[g] > 0i32 || return

    ∇ᵢ = ∇[i]
    ∇ᵢ² = ∇ᵢ^2

    μᵢ = μ[i] = β1 * μ[i] + (1f0 - β1) * ∇ᵢ
    νᵢ = ν[i] = β2 * ν[i] + (1f0 - β2) * ∇ᵢ²

    ωᵢ = Θ[i]
    Θ[i] = ωᵢ - lr * μᵢ / (√νᵢ + ϵ)
end
