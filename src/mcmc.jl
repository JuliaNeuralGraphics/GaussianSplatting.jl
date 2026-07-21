"""
Densification from "3D Gaussian Splatting as Markov Chain Monte Carlo"
(as implemented in LichtFeld Studio): instead of heuristic clone/split/prune
& opacity resets, the number of Gaussians only grows (up to `max_cap`) and
dead Gaussians (opacity ≤ `min_opacity`) are relocated onto alive ones with
an opacity/scale correction (Eq. 9) that preserves the render.
Position noise scaled by each Gaussian's covariance & opacity keeps the chain
exploring, while opacity & scale L1 regularization (see [`regularization_loss`](@ref))
provides the pressure that produces dead Gaussians to recycle.
"""
mutable struct MCMCStrategy <: AbstractStrategy
    # Eq. 9 coefficients: binoms[n, k + 1] = C(n-1, k)·(-1)^k/√(k+1).
    binoms::Matrix{Float32}

    max_cap::Int
    min_opacity::Float32
    start_refine::Int
    stop_refine::Int
    refine_every::Int
    grow_factor::Float32
    noise_lr::Float32
    opacity_reg::Float32
    scale_reg::Float32
    n_max::Int
end

function MCMCStrategy(;
    max_cap::Int = 1_000_000,
    min_opacity::Float32 = 0.005f0,
    start_refine::Int = 500,
    stop_refine::Int = 25_000,
    refine_every::Int = 100,
    grow_factor::Float32 = 1.05f0,
    noise_lr::Float32 = 5f5,
    opacity_reg::Float32 = 0.01f0,
    scale_reg::Float32 = 0.01f0,
    n_max::Int = 51,
)
    MCMCStrategy(
        mcmc_binom_coefficients(n_max),
        max_cap, min_opacity, start_refine, stop_refine, refine_every,
        grow_factor, noise_lr, opacity_reg, scale_reg, n_max)
end

function mcmc_binom_coefficients(n_max::Int)
    binoms = zeros(Float32, n_max, n_max)
    for n in 0:(n_max - 1)
        b = 1.0
        for k in 0:n
            sign = iseven(k) ? 1f0 : -1f0
            binoms[n + 1, k + 1] = Float32(b) * sign / sqrt(Float32(k + 1))
            k < n && (b *= (n - k) / (k + 1))
        end
    end
    return binoms
end

function regularization_loss(strategy::MCMCStrategy, opacities, scales)
    return strategy.opacity_reg * mean(NU.sigmoid.(opacities)) +
        strategy.scale_reg * mean(exp.(scales))
end

function post_train_step!(
    strategy::MCMCStrategy, gs::GaussianModel, optimizers,
    rast, camera::Camera, cache::GPUArrays.AllocCache;
    step::Int, extent::Float32,
)
    refining =
        strategy.start_refine < step < strategy.stop_refine &&
        step % strategy.refine_every == 0
    if refining
        GPUArrays.unsafe_free!(cache)
        relocate_gaussians!(strategy, gs, optimizers)
        check_finite(gs, "MCMC relocation")
        add_gaussians!(strategy, gs, optimizers)
        check_finite(gs, "MCMC growth")
    end
    inject_noise!(strategy, gs, optimizers.points.lr)
    isfinite(sum(gs.points)) || error("`points` are not finite after `MCMC noise injection`.")
    return
end

"""
Move dead Gaussians (opacity ≤ `min_opacity`) onto alive ones sampled with
probability ∝ opacity, correcting the target's opacity/scale (Eq. 9) so the
render is preserved. Adam moments of every touched Gaussian are reset.
"""
function relocate_gaussians!(strategy::MCMCStrategy, gs::GaussianModel, optimizers)
    o = Array(reshape(NU.sigmoid.(gs.opacities), :))
    dead = findall(≤(strategy.min_opacity), o)
    alive = findall(>(strategy.min_opacity), o)
    (isempty(dead) || isempty(alive)) && return 0

    ids = multinomial_sample(o[alive], length(dead))
    isempty(ids) && return 0
    sampled = alive[ids]

    kab = get_backend(gs)
    sampled_gpu = split_sampled!(strategy, gs, o, sampled)
    dead_gpu = adapt(kab, dead)

    # Copy relocated Gaussians onto the dead slots
    # (`dead ∩ sampled = ∅`, so gather-then-scatter is safe).
    gs.points[:, dead_gpu] = gs.points[:, sampled_gpu]
    gs.features_dc[:, :, dead_gpu] = gs.features_dc[:, :, sampled_gpu]
    if !isempty(gs.features_rest)
        gs.features_rest[:, :, dead_gpu] = gs.features_rest[:, :, sampled_gpu]
    end
    gs.scales[:, dead_gpu] = gs.scales[:, sampled_gpu]
    gs.rotations[:, dead_gpu] = gs.rotations[:, sampled_gpu]
    gs.opacities[:, dead_gpu] = gs.opacities[:, sampled_gpu]
    if gs.ids ≢ nothing
        gs.ids[dead_gpu] = gs.ids[sampled_gpu]
    end

    # Both source & destination rows received new parameters: reset their moments.
    touched = adapt(kab, union(sampled, dead))
    _zero_optimizer_rows!(optimizers.points, gs.points, touched)
    _zero_optimizer_rows!(optimizers.features_dc, gs.features_dc, touched)
    if !isempty(gs.features_rest)
        _zero_optimizer_rows!(optimizers.features_rest, gs.features_rest, touched)
    end
    _zero_optimizer_rows!(optimizers.scales, gs.scales, touched)
    _zero_optimizer_rows!(optimizers.rotations, gs.rotations, touched)
    _zero_optimizer_rows!(optimizers.opacities, gs.opacities, touched)

    KA.unsafe_free!(sampled_gpu)
    KA.unsafe_free!(dead_gpu)
    KA.unsafe_free!(touched)
    return length(dead)
end

"""
Grow the model by `grow_factor` (up to `max_cap`): sample sources ∝ opacity,
split them via Eq. 9 and append the copies (with zeroed Adam moments).
"""
function add_gaussians!(strategy::MCMCStrategy, gs::GaussianModel, optimizers)
    n = length(gs)
    n_new = min(strategy.max_cap, floor(Int, strategy.grow_factor * n)) - n
    n_new > 0 || return 0

    o = Array(reshape(NU.sigmoid.(gs.opacities), :))
    sampled = multinomial_sample(o, n_new)
    isempty(sampled) && return 0

    sampled_gpu = split_sampled!(strategy, gs, o, sampled)

    new_points = gs.points[:, sampled_gpu]
    new_features_dc = gs.features_dc[:, :, sampled_gpu]
    new_features_rest = isempty(gs.features_rest) ?
        gs.features_rest : gs.features_rest[:, :, sampled_gpu]
    new_scales = gs.scales[:, sampled_gpu]
    new_rotations = gs.rotations[:, sampled_gpu]
    new_opacities = gs.opacities[:, sampled_gpu]
    new_ids = gs.ids ≡ nothing ? nothing : gs.ids[sampled_gpu]

    append_gaussians!(gs, optimizers;
        new_points, new_features_dc, new_features_rest,
        new_scales, new_rotations, new_opacities, new_ids)

    KA.unsafe_free!(sampled_gpu)
    KA.unsafe_free!(new_points)
    KA.unsafe_free!(new_features_dc)
    KA.unsafe_free!(new_features_rest)
    KA.unsafe_free!(new_scales)
    KA.unsafe_free!(new_rotations)
    KA.unsafe_free!(new_opacities)
    isnothing(new_ids) || KA.unsafe_free!(new_ids)
    return n_new
end

# Sample `n` indices from `1:length(weights)` with probability ∝ `weights`.
function multinomial_sample(weights::AbstractVector{<:Real}, n::Int)
    cw = cumsum(Float64.(weights))
    total = cw[end]
    total > 0 || return Int[]
    return [min(searchsortedfirst(cw, rand() * total), length(cw)) for _ in 1:n]
end

"""
Recompute opacity & scale of each `sampled` Gaussian as if it is split into
`1 + multiplicity` identical copies (Eq. 9), scattering the results in-place.
`o` are the pre-update activated opacities. Returns `sampled` on the device.
"""
function split_sampled!(
    strategy::MCMCStrategy, gs::GaussianModel,
    o::Vector{Float32}, sampled::Vector{Int},
)
    kab = get_backend(gs)
    counts = zeros(Int32, length(o))
    for sid in sampled
        counts[sid] += 1
    end

    sampled_gpu = adapt(kab, sampled)
    s_old = Array(exp.(gs.scales[:, sampled_gpu]))

    new_o_raw = Matrix{Float32}(undef, 1, length(sampled))
    new_s_log = similar(s_old)
    for (i, sid) in enumerate(sampled)
        ratio = clamp(Int(counts[sid]) + 1, 1, strategy.n_max)
        new_o, coeff = relocation_params(strategy, o[sid], ratio)
        new_o_raw[1, i] = inverse_sigmoid(new_o)
        for j in 1:size(s_old, 1)
            new_s_log[j, i] = log(max(abs(coeff * s_old[j, i]), 1f-10))
        end
    end

    # Duplicate destinations write identical values, so the scatter is safe.
    gs.opacities[:, sampled_gpu] = adapt(kab, new_o_raw)
    gs.scales[:, sampled_gpu] = adapt(kab, new_s_log)
    return sampled_gpu
end

"""
Eq. 9 of the MCMC paper: opacity of each of the `ratio` identical copies that
together match one Gaussian with opacity `o`, and the scale multiplier `coeff`.
"""
function relocation_params(strategy::MCMCStrategy, o::Float32, ratio::Int)
    o = clamp(o, 1f-6, 1f0 - 1f-6)
    new_o = clamp(
        1f0 - (1f0 - o)^(1f0 / ratio),
        max(1f-6, strategy.min_opacity), 1f0 - 1f-6)

    denom = 0f0
    for i in 1:ratio, k in 0:(i - 1)
        denom += strategy.binoms[i, k + 1] * new_o^(k + 1)
    end
    # Sign-preserving floor of the denominator.
    denom = copysign(max(abs(denom), 1f-8), denom)
    coeff = clamp(o / denom, -1f6, 1f6)
    return new_o, coeff
end

"""
Perturb positions with noise `∝ Σ·ξ`, gated to near-dead Gaussians by a steep
opacity sigmoid & scaled by the (decaying) position learning rate.
"""
function inject_noise!(strategy::MCMCStrategy, gs::GaussianModel, points_lr::Float32)
    n = length(gs)
    n == 0 && return
    isotropic = size(gs.scales, 1) == 1
    _inject_noise!(get_backend(gs))(
        reinterpret(SVector{3, Float32}, gs.points),
        gs.opacities,
        isotropic ? gs.scales : reinterpret(SVector{3, Float32}, gs.scales),
        reinterpret(SVector{4, Float32}, gs.rotations),
        points_lr * strategy.noise_lr; ndrange=n)
    return
end

# TODO inbounds
@kernel cpu=false inbounds=false function _inject_noise!(
    points, opacities, scales, rotations, lr::Float32,
)
    i = @index(Global)
    ξ = SVector{3, Float32}(randn(Float32), randn(Float32), randn(Float32))

    R = unnorm_quat2rot(rotations[i])
    # Cap the variance: `exp` overflow (raw scale > 44) would give
    # `Σξ = ±Inf` & poison the position with `0·Inf = NaN`.
    s² = min.(exp.(2f0 .* scales[i]), 1f8) # Scalar for isotropic scales.
    # Σ·ξ = R·S²·Rᵀ·ξ.
    Σξ = R * (s² .* (R' * ξ))

    op = NU.sigmoid(opacities[i])
    # Cap the exponent: for opaque Gaussians (op ≳ 0.89) `exp` overflows
    # to `Inf` & `lr/Inf` relies on Inf-arithmetic to zero the factor.
    factor = lr / (1f0 + exp(min(100f0 * op - 0.5f0, 80f0)))
    points[i] = points[i] .+ factor .* Σξ
end
