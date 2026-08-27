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
    kind == :improved_gs ? ImprovedGSStrategy(gs; kwargs...) :
    kind == :default ? DefaultStrategy(gs; kwargs...) :
    kind == :mcmc ? MCMCStrategy(; kwargs...) :
    throw(ArgumentError("Unknown densification strategy `$kind`, must be one of: `:improved_gs`, `:default`, `:mcmc`."))

"""
Post train step every strategy must implement.

Besides `step` & `extent`, receives the `dataset` and a `view_target(idx)`
callable: a strategy may need views other than the one just trained on
(`ImprovedGSStrategy` re-renders a sample of them to score Gaussians).
Strategies that need neither take `kwargs...` instead.
"""
function post_train_step! end

"""
Fields of `strategy` holding per-gaussian statistics:
one entry per gaussian, indexed by gaussian id,
so they must be pruned & regrown in lockstep with the model.

`prune_points!` & `reset_stats!` maintain them generically.
"""
stat_array_names(::AbstractStrategy) = ()

"""
The `create_strategy` symbol that builds this strategy,
as a string — what the UI shows for "Strategy:".
"""
strategy_name(::AbstractStrategy) = "unknown"

# Keep only `valid_mask` of each per-gaussian statistic.
function prune_stats!(strategy::AbstractStrategy, valid_mask)
    for name in stat_array_names(strategy)
        x = getfield(strategy, name)
        new_x = x[valid_mask]
        KA.unsafe_free!(x)
        setfield!(strategy, name, new_x)
    end
    return
end

# Drop every per-gaussian statistic & reallocate it zeroed at length `n`.
function reset_stats!(strategy::AbstractStrategy, kab, n::Int)
    for name in stat_array_names(strategy)
        x = getfield(strategy, name)
        KA.unsafe_free!(x)
        setfield!(strategy, name, KA.zeros(kab, eltype(x), n))
    end
    return
end

include("default.jl")
include("mcmc.jl")
include("improved_gs.jl")
