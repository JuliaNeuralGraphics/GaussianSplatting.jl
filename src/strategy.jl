"""
A densification strategy: controls how the set of Gaussians grows/shrinks during training.
Each strategy owns its own state & hyper-parameters and implements [`post_train_step!`](@ref),
called once per train step after the optimizer update.
"""
abstract type AbstractStrategy end

"""
The original 3DGS adaptive density control: clone small/split big Gaussians
with high image-space positional gradient, prune transparent & oversized ones,
periodically reset opacity.
"""
mutable struct DefaultStrategy{
    R <: AbstractVector{Int32},
    G <: AbstractVector{Float32},
} <: AbstractStrategy
    # Per-Gaussian densification stats.
    max_radii::R
    accum_∇means_2d::G
    denom::G

    dense_percent::Float32
    densify_from_iter::Int
    densify_until_iter::Int
    densification_interval::Int
    densify_grad_threshold::Float32
    opacity_reset_interval::Int
    min_opacity::Float32
end

function DefaultStrategy(gs::GaussianModel;
    dense_percent::Float32 = 1f-2,
    densify_from_iter::Int = 500,
    densify_until_iter::Int = 15_000,
    densification_interval::Int = 100,
    densify_grad_threshold::Float32 = 2f-4,
    opacity_reset_interval::Int = 30_000,
    min_opacity::Float32 = 0.05f0,
)
    kab = get_backend(gs)
    n = length(gs)
    DefaultStrategy(
        KA.zeros(kab, Int32, n),
        KA.zeros(kab, Float32, n),
        KA.zeros(kab, Float32, n),
        dense_percent,
        densify_from_iter,
        densify_until_iter,
        densification_interval,
        densify_grad_threshold,
        opacity_reset_interval,
        min_opacity)
end

function post_train_step!(
    strategy::DefaultStrategy, gs::GaussianModel, optimizers,
    rast, camera::Camera, cache::GPUArrays.AllocCache;
    step::Int, extent::Float32,
)
    step ≤ strategy.densify_until_iter || return

    update_stats!(strategy, rast.gstate.radii,
        rast.gstate.∇means_2d, camera.intrinsics.resolution)

    do_densify =
        step ≥ strategy.densify_from_iter &&
        step % strategy.densification_interval == 0
    if do_densify
        GPUArrays.unsafe_free!(cache)

        max_screen_size::Int32 =
            step > strategy.opacity_reset_interval ? 20 : 0
        densify_and_prune!(strategy, gs, optimizers;
            extent, pruning_extent=extent, max_screen_size)
    end

    if step % strategy.opacity_reset_interval == 0
        reset_opacity!(gs)
        NU.reset!(optimizers.opacities)
    end
    return
end

function update_stats!(
    strategy::DefaultStrategy, radii::AbstractVector{Int32},
    ∇means_2d::AbstractVector{SVector{2, Float32}},
    resolution::SVector{2, UInt32},
)
    _update_stats!(get_backend(strategy.max_radii), 256)(
        strategy.max_radii, strategy.accum_∇means_2d, strategy.denom,
        radii, ∇means_2d, resolution; ndrange=length(strategy.max_radii))
    return
end

@kernel cpu=false inbounds=true function _update_stats!(
    # Outputs.
    max_radii::AbstractVector{Int32},
    accum_∇means_2d::AbstractVector{Float32},
    denom::AbstractVector{Float32},
    # Inputs.
    radii::AbstractVector{Int32},
    ∇means_2d::AbstractVector{SVector{2, Float32}},
    resolution::SVector{2, UInt32},
)
    i = @index(Global)
    r = radii[i]
    r > 0 || return

    max_radii[i] = max(max_radii[i], r)
    ∇mean_2d = ∇means_2d[i] .* resolution .* 0.5f0
    accum_∇means_2d[i] += norm(∇mean_2d)
    denom[i] += 1f0
end
