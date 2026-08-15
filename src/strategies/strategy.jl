"""
A densification strategy: controls how the set of Gaussians grows/shrinks during training.
Each strategy owns its own state & hyper-parameters and implements [`post_train_step!`](@ref),
called once per train step after the optimizer update.
"""
abstract type AbstractStrategy end

"""
Strategy-specific loss term added to the photometric loss in `step!`.
Must be Zygote-differentiable w.r.t. `opacities` & `scales` (raw, pre-activation).
"""
regularization_loss(::AbstractStrategy, opacities, scales) = 0f0

create_strategy(kind::Symbol, gs; kwargs...) =
    kind == :default ? DefaultStrategy(gs; kwargs...) :
    kind == :mcmc ? MCMCStrategy(; kwargs...) :
    throw(ArgumentError("Unknown densification strategy `$kind`, must be one of: `:default`, `:mcmc`."))

"""
Post train step every strategy must implement.
"""
function post_train_step! end

include("default.jl")
include("mcmc.jl")
