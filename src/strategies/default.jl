"""
The original 3DGS adaptive density control: clone small/split big Gaussians
with high image-space positional gradient, prune transparent & oversized ones,
periodically reset opacity.
"""
mutable struct DefaultStrategy{
    G <: AbstractVector{Float32},
} <: AbstractStrategy
    # Per-Gaussian densification stats.
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
    opacity_reset_interval::Int = 3_000,
    min_opacity::Float32 = 0.005f0,
)
    kab = get_backend(gs)
    n = length(gs)
    DefaultStrategy(
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

strategy_name(::DefaultStrategy) = "default"
stat_array_names(::DefaultStrategy) = (:accum_∇means_2d, :denom)

memory_usage(strategy::DefaultStrategy) =
    sum(n -> memory_usage(getfield(strategy, n)), stat_array_names(strategy); init=0)

function KA.unsafe_free!(strategy::DefaultStrategy)
    for name in stat_array_names(strategy)
        KA.unsafe_free!(getfield(strategy, name))
    end
    return
end

function post_train_step!(
    strategy::DefaultStrategy, gs::GaussianModel, optimizers,
    rast, camera::Camera, cache::GPUArrays.AllocCache;
    step::Int, extent::Float32, kwargs...,
)
    step ≤ strategy.densify_until_iter || return

    update_stats!(strategy, rast.gstate.radii,
        rast.gstate.∇means_2d, camera.intrinsics.resolution)

    do_densify =
        step ≥ strategy.densify_from_iter &&
        step % strategy.densification_interval == 0
    if do_densify
        GPUArrays.unsafe_free!(cache)
        densify_and_prune!(strategy, gs, optimizers; extent,
            pruning_extent=extent,
            prune_big=step > strategy.opacity_reset_interval)
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
    _update_stats!(get_backend(strategy.denom), 256)(
        strategy.accum_∇means_2d, strategy.denom,
        radii, ∇means_2d, resolution; ndrange=length(strategy.denom))
    return
end

@kernel cpu=false inbounds=true function _update_stats!(
    # Outputs.
    accum_∇means_2d::AbstractVector{Float32},
    denom::AbstractVector{Float32},
    # Inputs.
    radii::AbstractVector{Int32},
    ∇means_2d::AbstractVector{SVector{2, Float32}},
    resolution::SVector{2, UInt32},
)
    i = @index(Global)
    radii[i] > 0 || return

    ∇mean_2d = ∇means_2d[i] .* resolution .* 0.5f0
    accum_∇means_2d[i] += norm(∇mean_2d)
    denom[i] += 1f0
end

function densify_and_prune!(
    strategy::DefaultStrategy, gs::GaussianModel, optimizers;
    extent::Float32, pruning_extent::Float32, prune_big::Bool,
)
    grad_threshold = strategy.densify_grad_threshold
    dense_percent = strategy.dense_percent

    ∇means_2d = strategy.accum_∇means_2d ./ strategy.denom
    mask = isnan.(∇means_2d)
    ∇means_2d[mask] .= 0f0
    KA.unsafe_free!(mask)

    densify_clone!(strategy, gs, optimizers; ∇means_2d, grad_threshold, extent, dense_percent)
    densify_split!(strategy, gs, optimizers; ∇means_2d, grad_threshold, extent, dense_percent)
    KA.unsafe_free!(∇means_2d)

    # Prune points that are too transparent and (once opacity has been reset at least once)
    # those with a high scale in world space.
    valid_mask = reshape(NU.sigmoid.(gs.opacities) .> strategy.min_opacity, :)
    if prune_big
        γ = 0.1f0 * pruning_extent
        valid_mask .&= reshape(maximum(exp.(gs.scales); dims=1) .< γ, :)
    end
    prune_points!(strategy, gs, optimizers, valid_mask)
    return
end

function densify_clone!(strategy::DefaultStrategy, gs::GaussianModel, optimizers;
    ∇means_2d, grad_threshold::Float32,
    extent::Float32, dense_percent::Float32,
)
    # Clone gaussians that have high gradient and small size.
    γ = extent * dense_percent
    mask =
        ∇means_2d .> grad_threshold .&&
        reshape(maximum(exp.(gs.scales); dims=1) .< γ, :)

    new_points = gs.points[:, mask]
    new_features_dc = gs.features_dc[:, :, mask]
    new_features_rest = isempty(gs.features_rest) ? gs.features_rest : gs.features_rest[:, :, mask]
    new_scales = gs.scales[:, mask]
    new_rotations = gs.rotations[:, mask]
    new_opacities = gs.opacities[:, mask]

    new_ids = gs.ids ≡ nothing ? nothing : gs.ids[mask]
    KA.unsafe_free!(mask)

    densification_postfix!(strategy, gs, optimizers;
        new_points, new_features_dc, new_features_rest,
        new_scales, new_rotations, new_opacities, new_ids)

    KA.unsafe_free!(new_points)
    KA.unsafe_free!(new_features_dc)
    KA.unsafe_free!(new_features_rest)
    KA.unsafe_free!(new_scales)
    KA.unsafe_free!(new_rotations)
    KA.unsafe_free!(new_opacities)
    isnothing(new_ids) || KA.unsafe_free!(new_ids)
    return
end

function densify_split!(strategy::DefaultStrategy, gs::GaussianModel, optimizers;
    ∇means_2d, grad_threshold::Float32,
    extent::Float32, dense_percent::Float32,
)
    kab = get_backend(gs)
    n = size(gs.points, 2)
    n_split = 2

    padded_grad = KA.zeros(kab, eltype(∇means_2d), n)
    padded_grad[1:length(∇means_2d)] .= ∇means_2d

    # Split gaussians that have high gradient and big size.
    γ = extent * dense_percent
    mask =
        padded_grad .≥ grad_threshold .&&
        reshape(maximum(exp.(gs.scales); dims=1) .> γ, :)
    stds = repeat(exp.(gs.scales)[:, mask], 1, n_split)

    new_points = repeat(gs.points[:, mask], 1, n_split)
    new_features_rest = isempty(gs.features_rest) ?
        gs.features_rest : repeat(gs.features_rest[:, :, mask], 1, 1, n_split)
    new_features_dc = repeat(gs.features_dc[:, :, mask], 1, 1, n_split)
    new_scales = log.(stds ./ (0.8f0 * n_split))
    new_rotations = repeat(gs.rotations[:, mask], 1, n_split)
    new_opacities = repeat(gs.opacities[:, mask], 1, n_split)

    new_ids = gs.ids ≡ nothing ? nothing : repeat(gs.ids[mask], n_split)

    n_new_points = size(new_points, 2)
    if n_new_points > 0
        isotropic = size(gs.scales, 1) == 1
        _add_split_noise!(kab)(
            reinterpret(SVector{3, Float32}, new_points),
            reinterpret(SVector{4, Float32}, new_rotations),
            isotropic ? stds : reinterpret(SVector{3, Float32}, stds);
            ndrange=n_new_points)
    end
    KA.unsafe_free!(stds)

    densification_postfix!(strategy, gs, optimizers;
        new_points, new_features_dc, new_features_rest,
        new_scales, new_rotations, new_opacities, new_ids)

    KA.unsafe_free!(new_points)
    KA.unsafe_free!(new_features_dc)
    KA.unsafe_free!(new_features_rest)
    KA.unsafe_free!(new_scales)
    KA.unsafe_free!(new_rotations)
    KA.unsafe_free!(new_opacities)
    isnothing(new_ids) || KA.unsafe_free!(new_ids)

    # Prune gaussians that have small gradient or small size
    # ignoring newly inserted gaussians.
    valid_mask = vcat(.!mask, KA.ones(kab, Bool, n_new_points))
    KA.unsafe_free!(mask)
    prune_points!(strategy, gs, optimizers, valid_mask)
    return
end

@kernel cpu=false inbounds=true function _add_split_noise!(points, rots, stds)
    i = @index(Global)
    σ = stds[i]
    ξ = σ .* SVector{3, Float32}(randn(Float32), randn(Float32), randn(Float32))

    q = rots[i]
    R = unnorm_quat2rot(q)
    p = points[i]
    points[i] = p .+ R * ξ
end

function prune_points!(strategy::AbstractStrategy, gs::GaussianModel, optimizers, valid_mask)
    _prune_optimizer!(optimizers.points, valid_mask, gs.points)
    new_points = gs.points[:, valid_mask]
    KA.unsafe_free!(gs.points)
    gs.points = new_points

    _prune_optimizer!(optimizers.features_dc, valid_mask, gs.features_dc)
    new_features_dc = gs.features_dc[:, :, valid_mask]
    KA.unsafe_free!(gs.features_dc)
    gs.features_dc = new_features_dc

    if !isempty(gs.features_rest)
        _prune_optimizer!(optimizers.features_rest, valid_mask, gs.features_rest)
        new_features_rest = gs.features_rest[:, :, valid_mask]
        KA.unsafe_free!(gs.features_rest)
        gs.features_rest = new_features_rest
    end

    _prune_optimizer!(optimizers.scales, valid_mask, gs.scales)
    new_scales = gs.scales[:, valid_mask]
    KA.unsafe_free!(gs.scales)
    gs.scales = new_scales

    _prune_optimizer!(optimizers.rotations, valid_mask, gs.rotations)
    new_rotations = gs.rotations[:, valid_mask]
    KA.unsafe_free!(gs.rotations)
    gs.rotations = new_rotations

    _prune_optimizer!(optimizers.opacities, valid_mask, gs.opacities)
    new_opacities = gs.opacities[:, valid_mask]
    KA.unsafe_free!(gs.opacities)
    gs.opacities = new_opacities

    prune_stats!(strategy, valid_mask)

    if gs.ids ≢ nothing
        new_ids = gs.ids[valid_mask]
        KA.unsafe_free!(gs.ids)
        gs.ids = new_ids
    end
    return
end

"""
Append the new Gaussians, then drop the accumulated per-gaussian statistics:
they describe the pre-densification set & there is no meaningful value to carry
over to the newly inserted rows.
"""
function densification_postfix!(
    strategy::AbstractStrategy, gs::GaussianModel, optimizers;
    new_points, new_features_dc, new_features_rest,
    new_scales, new_rotations, new_opacities, new_ids,
)
    append_gaussians!(gs, optimizers;
        new_points, new_features_dc, new_features_rest,
        new_scales, new_rotations, new_opacities, new_ids)
    reset_stats!(strategy, get_backend(gs), size(gs.points, 2))
    return
end

# Append new Gaussians to the model & extend optimizer states
# with zeroed moments for the new rows.
function append_gaussians!(
    gs::GaussianModel, optimizers;
    new_points, new_features_dc, new_features_rest,
    new_scales, new_rotations, new_opacities, new_ids,
)
    _append_optimizer!(optimizers.points, new_points)
    new_points = cat(gs.points, new_points; dims=ndims(new_points))
    KA.unsafe_free!(gs.points)
    gs.points = new_points

    _append_optimizer!(optimizers.features_dc, new_features_dc)
    new_features_dc = cat(gs.features_dc, new_features_dc; dims=ndims(new_features_dc))
    KA.unsafe_free!(gs.features_dc)
    gs.features_dc = new_features_dc

    if !isempty(gs.features_rest)
        _append_optimizer!(optimizers.features_rest, new_features_rest)
        new_features_rest = cat(gs.features_rest, new_features_rest; dims=ndims(new_features_rest))
        KA.unsafe_free!(gs.features_rest)
        gs.features_rest = new_features_rest
    end

    _append_optimizer!(optimizers.scales, new_scales)
    new_scales = cat(gs.scales, new_scales; dims=ndims(new_scales))
    KA.unsafe_free!(gs.scales)
    gs.scales = new_scales

    _append_optimizer!(optimizers.rotations, new_rotations)
    new_rotations = cat(gs.rotations, new_rotations; dims=ndims(new_rotations))
    KA.unsafe_free!(gs.rotations)
    gs.rotations = new_rotations

    _append_optimizer!(optimizers.opacities, new_opacities)
    new_opacities = cat(gs.opacities, new_opacities; dims=ndims(new_opacities))
    KA.unsafe_free!(gs.opacities)
    gs.opacities = new_opacities

    if gs.ids ≢ nothing
        new_ids = cat(gs.ids, new_ids; dims=1)
        KA.unsafe_free!(gs.ids)
        gs.ids = new_ids
    end
    return
end

function _append_optimizer!(opt::NU.Adam, extension)
    kab = get_backend(extension)
    dims = ndims(extension)

    μ = opt.μ[1]
    μ̂ = KA.zeros(kab, eltype(μ), size(extension))
    μ = cat(reshape(μ, size(extension)[1:end - 1]..., :), μ̂; dims)
    KA.unsafe_free!(opt.μ[1])
    opt.μ[1] = reshape(μ, :)

    ν = opt.ν[1]
    ν̂ = KA.zeros(kab, eltype(ν), size(extension))
    ν = cat(reshape(ν, size(extension)[1:end - 1]..., :), ν̂; dims)
    KA.unsafe_free!(opt.ν[1])
    opt.ν[1] = reshape(ν, :)
    return
end

function _prune_optimizer!(opt::NU.Adam, mask, x)
    d = ntuple(i -> Colon(), ndims(x) - 1)
    new_μ = reshape(reshape(opt.μ[1], size(x))[d..., mask], :)
    new_ν = reshape(reshape(opt.ν[1], size(x))[d..., mask], :)
    KA.unsafe_free!(opt.μ[1])
    KA.unsafe_free!(opt.ν[1])
    opt.μ[1] = new_μ
    opt.ν[1] = new_ν
    return
end

# Zero Adam moments for Gaussians at `idxs` (parameters shaped as `x`),
# e.g. after their parameters were replaced by relocation.
function _zero_optimizer_rows!(opt::NU.Adam, x, idxs)
    d = ntuple(i -> Colon(), ndims(x) - 1)
    reshape(opt.μ[1], size(x))[d..., idxs] .= 0f0
    reshape(opt.ν[1], size(x))[d..., idxs] .= 0f0
    return
end
