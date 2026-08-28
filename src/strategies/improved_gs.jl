"""
ImprovedGS — "Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering"
(Deng et al., arXiv:2508.12313).

Four changes to the 3DGS adaptive density control, aiming for higher fidelity at
a *lower* final Gaussian count:

- **Edge-Aware Score (EAS)**: which Gaussians to split is decided from how much
  image *detail* they cover, `Sᵢⱼ = Σₚ ωₚ·αᵢₚ·Tᵢₚ` for view `j`, where `ω` is the
  Canny edge magnitude of the ground-truth image.
  Computed at each refine by re-rendering a sample of train views forward-only
  (see [`edge_scores`](@ref)); `render!` accumulates `S` as it blends, since the
  alpha-compositing weight is already in hand there.

- **Absolute gradient criterion (AbsGS)**: candidates are Gaussians whose mean
  *absolute* image-space positional gradient `Σₚ |∂L/∂μ|` exceeds a threshold.
  The signed sum `DefaultStrategy` uses cancels itself out for a Gaussian
  straddling an edge, which is exactly the case that needs splitting.
  Requires `enable_abs_grad!` on the rasterizer.

- **Long-Axis Split (LAS)** replaces 3DGS's clone-and-split: a parent becomes
  two Gaussians offset along its longest principal axis, shrunk so the children
  stay inside the parent's ellipsoid.
  No random jitter, so densification does not perturb geometry that is already correct.

- **Growth Control (GC)**: the total count follows an explicit `√` schedule
  from `start_refine` to `stop_refine`, rather than being whatever the gradient
  threshold happens to produce. `max_cap` is a target, not just a ceiling.

- **Recovery-Aware Pruning (RAP)**: some iterations after an opacity reset,
  the Gaussians that recovered least are the overfitted ones — prune the bottom
  `recovery_prune_fraction` by opacity at `recovery_prune_iters`.

Unlike `DefaultStrategy`, opacity is the only prune criterion:
there is no screen-size or world-scale prune.
Growth Control bounds the population directly, which is what those terms were compensating for.

## Divergences from the official implementation

Ported against LichtFeld-Studio's `improved_gs_plus.cpp`. The authors' own code
(github.com/XiaoBin2001/Improved-GS) differs in the places below; they are
listed so the gap is explicit rather than accidental.
The split geometry is the substantive one.

| | Official (= paper) | Here (= LichtFeld) |
|---|---|---|
| split offset | `1.35σ` (`3σ · 0.45`, i.e. `L₀` is the *3σ* semi-axis) | `0.5σ` |
| long-axis scale | `×0.55` (`1 - d`) | `×0.5` |
| short-axis scale | `×0.8930` (`√(1 - d²)`) | `×0.85` |
| opacity factor | `×0.6` | `×0.6` |
| grad threshold | `3f-4` | `2f-4` |
| budget ramp ends at | `stop_refine - 500` | `stop_refine` |
| edge operator | `FIND_EDGES` 3×3 on uint8 luminance, min-max normalized | Canny + NMS, median-normalized |
| EAS view pool | popped from a fixed list; every image resident on the GPU | reshuffled each sweep; images re-read from disk (see [`edge_scores`](@ref)) |
| refine interval | `100` | `500` |
| late training | past `stop_refine - 100`: threshold `÷1.5` & score falls back to the gradient | not implemented |
| one-off prune at iter 300 | `only_prune(0.02)` | not implemented |

Two places where this implementation deliberately does *not* follow the
official code, both documented at the function concerned:
[`recovery_prune!`](@ref) prunes an exact count instead of thresholding on the
quantile value (which collapses on the opacity ties a reset creates), and
[`select_split_indices`](@ref) draws via Gumbel-top-k, which is equivalent to
the official `multinomial(replacement=false)`.

The paper's fifth component, Multi-step Update, is not implemented: it is a
training-loop gradient-accumulation change rather than a densification one.

## Supervision this repo has & the reference does not

Coverage masks & sky masks scope the edge score to the region the loss actually
supervises (see [`supervision_weight`](@ref)). Depth priors and the geometry
regularizers need no handling: they are loss terms, so they reach densification
through the AbsGS gradient like every other term.
"""
mutable struct ImprovedGSStrategy{
    G <: AbstractVector{Float32},
    M <: AbstractMatrix{Float32},
    V <: AbstractMatrix{SVector{2, Float32}},
} <: AbstractStrategy
    # Per-Gaussian densification stats, accumulated over one refine window.
    accum_∇means_2d_abs::G
    denom::G

    # Canny scratch, sized to the current view resolution & reallocated when
    # it changes (the dataset may serve views at differing resolutions).
    edge_map::M   # NMS output, median-normalized: the `ω` handed to `render!`.
    edge_blur::M  # Luminance after the 5×5 Gaussian.
    edge_grad::V  # Sobel (∂x, ∂y) of `edge_blur`.
    # Per-view positive-median of the raw NMS output, keyed by dataset view id.
    # The ground truth never changes, so this is computed once per view instead
    # of re-sorting the edge map on every step.
    edge_medians::Dict{Int, Float32}
    # Views not yet drawn by the current sweep of the edge-score pass, in the
    # order they will be taken. Refilled (reshuffled) when it runs out, so a
    # sweep covers every view before any is scored twice.
    view_pool::Vector{Int}

    max_cap::Int
    densify_grad_threshold::Float32
    start_refine::Int
    stop_refine::Int
    refine_every::Int

    # LAS. Offset is `split_offset · exp(s_max)`, i.e. a fraction of the parent's
    # 1σ semi-axis; the long axis is scaled by `split_scale_long` & the two short
    # ones by `split_scale_short`. These are LichtFeld-Studio's constants — see
    # the divergence table above for the official ones, which put the children
    # considerably further apart.
    split_offset::Float32
    split_scale_long::Float32
    split_scale_short::Float32
    split_opacity_factor::Float32

    opacity_reset_interval::Int
    opacity_reset_value::Float32
    recovery_prune_iters::Vector{Int}
    recovery_prune_fraction::Float32

    min_opacity::Float32
    # Edge magnitudes below this are the floating-point noise floor of
    # convolving a flat region, not edges. See `_canny_nms!`.
    edge_min::Float32

    # Views scored per refine, & the refines that score *every* view instead.
    edge_sample_cams::Int
    edge_full_scan_iters::Vector{Int}
end

function ImprovedGSStrategy(gs::GaussianModel;
    max_cap::Int = 1_000_000,
    densify_grad_threshold::Float32 = 2f-4,
    start_refine::Int = 500,
    stop_refine::Int = 15_000,
    refine_every::Int = 500,
    split_offset::Float32 = 0.5f0,
    split_scale_long::Float32 = 0.5f0,
    split_scale_short::Float32 = 0.85f0,
    split_opacity_factor::Float32 = 0.6f0,
    opacity_reset_interval::Int = 3_000,
    opacity_reset_value::Float32 = 0.05f0,
    # 300 iterations after each of the first two opacity resets. Kept inside the
    # early densification phase so the final count is not disturbed.
    recovery_prune_iters::Vector{Int} = [k * opacity_reset_interval + 300 for k in 1:2],
    recovery_prune_fraction::Float32 = 0.2f0,
    min_opacity::Float32 = 0.005f0,
    edge_min::Float32 = 1f-4,
    edge_sample_cams::Int = 10,
    # 400 iterations after each of the first two opacity resets, rounded up to a
    # refine step so it actually lands on one. `Int[]` disables it: a full scan
    # reads & renders the entire train set, which is minutes on a lazily loaded
    # dataset (the reference keeps every image resident on the GPU).
    edge_full_scan_iters::Vector{Int} = [
        cld(k * opacity_reset_interval + 400, refine_every) * refine_every
        for k in 1:2],
)
    kab = get_backend(gs)
    n = length(gs)
    ImprovedGSStrategy(
        KA.zeros(kab, Float32, n),
        KA.zeros(kab, Float32, n),

        KA.zeros(kab, Float32, (0, 0)),
        KA.zeros(kab, Float32, (0, 0)),
        KA.zeros(kab, SVector{2, Float32}, (0, 0)),
        Dict{Int, Float32}(),
        Int[],

        max_cap,
        densify_grad_threshold,
        start_refine,
        stop_refine,
        refine_every,
        split_offset,
        split_scale_long,
        split_scale_short,
        split_opacity_factor,
        opacity_reset_interval,
        opacity_reset_value,
        recovery_prune_iters,
        recovery_prune_fraction,
        min_opacity,
        edge_min,
        edge_sample_cams,
        edge_full_scan_iters)
end

strategy_name(::ImprovedGSStrategy) = "improved_gs"

stat_array_names(::ImprovedGSStrategy) = (:accum_∇means_2d_abs, :denom)

memory_usage(strategy::ImprovedGSStrategy) =
    sum(n -> memory_usage(getfield(strategy, n)), stat_array_names(strategy); init=0) +
    memory_usage(strategy.edge_map) +
    memory_usage(strategy.edge_blur) +
    memory_usage(strategy.edge_grad)

function KA.unsafe_free!(strategy::ImprovedGSStrategy)
    for name in stat_array_names(strategy)
        KA.unsafe_free!(getfield(strategy, name))
    end
    KA.unsafe_free!(strategy.edge_map)
    KA.unsafe_free!(strategy.edge_blur)
    KA.unsafe_free!(strategy.edge_grad)
    return
end

function post_train_step!(
    strategy::ImprovedGSStrategy, gs::GaussianModel, optimizers,
    rast, camera::Camera, cache::GPUArrays.AllocCache;
    step::Int, extent::Float32,
    dataset::Maybe{ColmapDataset} = nothing, view_target = nothing,
)
    step ≤ strategy.stop_refine || return

    update_stats!(strategy, rast.gstate.radii,
        rast.gstate.∇means_2d_abs, camera.intrinsics.resolution)

    if step in strategy.recovery_prune_iters
        KA.synchronize(get_backend(gs))
        GPUArrays.unsafe_free!(cache)
        recovery_prune!(strategy, gs, optimizers)
    end

    do_densify =
        step ≥ strategy.start_refine &&
        step % strategy.refine_every == 0
    if do_densify
        KA.synchronize(get_backend(gs))
        GPUArrays.unsafe_free!(cache)
        densify_and_prune!(strategy, gs, optimizers, rast, dataset; step, view_target)
    end

    if step % strategy.opacity_reset_interval == 0
        reset_opacity!(gs, strategy.opacity_reset_value)
        NU.reset!(optimizers.opacities)
    end
    return
end

function update_stats!(
    strategy::ImprovedGSStrategy, radii::AbstractVector{Int32},
    ∇means_2d_abs::AbstractVector{SVector{2, Float32}},
    resolution::SVector{2, UInt32},
)
    _update_abs_stats!(get_backend(strategy.denom), 256)(
        strategy.accum_∇means_2d_abs, strategy.denom,
        radii, ∇means_2d_abs, resolution; ndrange=length(strategy.denom))
    return
end

@kernel cpu=false inbounds=true function _update_abs_stats!(
    # Outputs.
    accum_∇means_2d_abs::AbstractVector{Float32},
    denom::AbstractVector{Float32},
    # Inputs.
    radii::AbstractVector{Int32},
    ∇means_2d_abs::AbstractVector{SVector{2, Float32}},
    resolution::SVector{2, UInt32},
)
    i = @index(Global)
    radii[i] > 0 || return

    # NDC -> pixel rescaling.
    ∇mean_2d = ∇means_2d_abs[i] .* resolution .* 0.5f0
    accum_∇means_2d_abs[i] += norm(∇mean_2d)
    denom[i] += 1f0
end

"""
Target Gaussian count at `step`: `N_max·√((I - I_start) / (I_end - I_start))`.

A `√` rather than a straight ramp, so most of the budget is spent early
(where splitting still has time to be optimized)
and the peak count is only reached at the very end of the densification phase.
"""
function growth_budget(strategy::ImprovedGSStrategy, step::Int)
    span = strategy.stop_refine - strategy.start_refine
    span > 0 || return strategy.max_cap
    t = clamp((step - strategy.start_refine) / span, 0f0, 1f0)
    return round(Int, strategy.max_cap * sqrt(t))
end

function densify_and_prune!(
    strategy::ImprovedGSStrategy, gs::GaussianModel, optimizers,
    rast = nothing, dataset = nothing; step::Int, view_target = nothing,
)
    n_new = growth_budget(strategy, step) - length(gs)
    if n_new > 0
        # Read the window's statistics before anything mutates them.
        # `denom == 0` for Gaussians no view rendered this window:
        # the divisions give NaN, which `select_split_indices` filters out.
        ∇means_2d_abs = strategy.accum_∇means_2d_abs ./ strategy.denom
        edge = edge_scores(strategy, gs, rast, dataset; step, view_target)

        idxs = select_split_indices(strategy, ∇means_2d_abs, edge, n_new)
        KA.unsafe_free!(∇means_2d_abs)
        edge ≡ nothing || KA.unsafe_free!(edge)

        isempty(idxs) || long_axis_split!(strategy, gs, optimizers, idxs)
    end

    # Opacity is the *only* prune criterion.
    valid_mask = reshape(NU.sigmoid.(gs.opacities) .> strategy.min_opacity, :)
    prune_points!(strategy, gs, optimizers, valid_mask)
    KA.unsafe_free!(valid_mask)

    # Statistics are per-window: start the next one clean.
    reset_stats!(strategy, get_backend(gs), length(gs))
    return
end

"""
Pick `n_new` Gaussians to split:
candidates are those whose mean absolute image-space gradient is bigger than
`densify_grad_threshold`, sampled without replacement with probability
proportional to their edge-aware score (uniformly when `edge ≡ nothing`).

Sampling uses the Gumbel-top-k trick — adding `-log(-log u)` to each log-weight
and taking the `k` largest keys draws exactly a weighted sample without replacement.
Runs on the host, like `MCMCStrategy`'s `multinomial_sample`:
a few MB of transfer, once per refine.
"""
function select_split_indices(strategy::ImprovedGSStrategy, ∇means_2d_abs, edge, n_new::Int)
    n_new > 0 || return Int[]

    g = Array(∇means_2d_abs)
    candidates = findall(x -> isfinite(x) && x > strategy.densify_grad_threshold, g)
    isempty(candidates) && return Int[]
    length(candidates) ≤ n_new && return candidates

    w = if edge ≡ nothing
        ones(length(candidates))
    else
        e = Array(edge)
        [(isfinite(e[i]) && e[i] > 0f0) ? Float64(e[i]) : 0.0 for i in candidates]
    end
    # No edge signal at all (e.g. a textureless scene):
    # fall back to a uniform draw over the candidates rather than degenerating to the first `n_new`.
    all(iszero, w) && fill!(w, 1.0)

    keys = similar(w)
    for j in eachindex(w)
        # `rand()` may return exactly 0; clamp so the Gumbel term stays finite.
        u = max(rand(), 1e-12)
        keys[j] = log(w[j] + 1e-12) - log(-log(u))
    end
    return candidates[partialsortperm(keys, 1:n_new; rev=true)]
end

"""
Split each Gaussian in `idxs` along its longest principal axis:
the parent is moved `+δ` and rewritten in place, one child is appended at `-δ`,
and both are shrunk & faded so the pair approximates the original.

With raw log-scales `s`, longest axis `l` and rotation `R`:

    δ         = R[:, l] · split_offset · exp(s[l])
    s'[l]     = s[l] + log(split_scale_long)
    s'[other] = s[other] + log(split_scale_short)
    opacity'  = logit(split_opacity_factor · sigmoid(opacity))

Rotation and every SH coefficient are inherited unchanged.
"""
function long_axis_split!(strategy::ImprovedGSStrategy, gs::GaussianModel, optimizers, idxs::Vector{Int})
    kab = get_backend(gs)
    idxs_gpu = adapt(kab, idxs)

    # Snapshot the parents *before* the kernel mutates them: the children start
    # as exact copies and the kernel then overwrites their geometry.
    new_points = gs.points[:, idxs_gpu]
    new_features_dc = gs.features_dc[:, :, idxs_gpu]
    new_features_rest = isempty(gs.features_rest) ?
        gs.features_rest : gs.features_rest[:, :, idxs_gpu]
    new_scales = gs.scales[:, idxs_gpu]
    new_rotations = gs.rotations[:, idxs_gpu]
    new_opacities = gs.opacities[:, idxs_gpu]
    new_ids = gs.ids ≡ nothing ? nothing : gs.ids[idxs_gpu]

    _long_axis_split!(kab)(
        gs.points, gs.scales, gs.opacities,
        new_points, new_scales, new_opacities,
        gs.rotations, idxs_gpu,
        strategy.split_offset, strategy.split_scale_long,
        strategy.split_scale_short, strategy.split_opacity_factor;
        ndrange=length(idxs))

    densification_postfix!(strategy, gs, optimizers;
        new_points, new_features_dc, new_features_rest,
        new_scales, new_rotations, new_opacities, new_ids)

    # The parents now hold different parameters than the moments describe;
    # the children get zeroed moments from `_append_optimizer!` already.
    _zero_optimizer_rows!(optimizers.points, gs.points, idxs_gpu)
    _zero_optimizer_rows!(optimizers.features_dc, gs.features_dc, idxs_gpu)
    isempty(gs.features_rest) || _zero_optimizer_rows!(optimizers.features_rest, gs.features_rest, idxs_gpu)
    _zero_optimizer_rows!(optimizers.scales, gs.scales, idxs_gpu)
    _zero_optimizer_rows!(optimizers.rotations, gs.rotations, idxs_gpu)
    _zero_optimizer_rows!(optimizers.opacities, gs.opacities, idxs_gpu)

    KA.unsafe_free!(idxs_gpu)
    KA.unsafe_free!(new_points)
    KA.unsafe_free!(new_features_dc)
    KA.unsafe_free!(new_features_rest)
    KA.unsafe_free!(new_scales)
    KA.unsafe_free!(new_rotations)
    KA.unsafe_free!(new_opacities)
    isnothing(new_ids) || KA.unsafe_free!(new_ids)
    return length(idxs)
end

@kernel cpu=false inbounds=true function _long_axis_split!(
    # Parents, modified in place.
    points::AbstractMatrix{Float32},
    scales::AbstractMatrix{Float32},
    opacities::AbstractMatrix{Float32},
    # Children, entering as copies of their parents.
    child_points::AbstractMatrix{Float32},
    child_scales::AbstractMatrix{Float32},
    child_opacities::AbstractMatrix{Float32},
    # Inputs.
    @Const(rotations), @Const(idxs),
    offset_factor::Float32, long_factor::Float32,
    short_factor::Float32, opacity_factor::Float32,
)
    j = @index(Global)
    i = idxs[j]
    d = size(scales, 1)

    # Longest principal axis. Isotropic models (`d == 1`) store a single shared
    # scale, so there is no long axis to find: split along the local x axis and
    # shrink uniformly by the short-axis factor, which matches the offset.
    l = 1
    for c in 2:d
        scales[c, i] > scales[l, i] && (l = c)
    end
    factor_long = d == 3 ? long_factor : short_factor

    q = SVector{4, Float32}(
        rotations[1, i], rotations[2, i], rotations[3, i], rotations[4, i])
    R = unnorm_quat2rot(q)
    magnitude = offset_factor * exp(scales[l, i])
    δ = SVector{3, Float32}(R[1, l], R[2, l], R[3, l]) .* magnitude

    # TODO unroll
    for c in 1:3
        p = points[c, i]
        points[c, i] = p + δ[c]
        child_points[c, j] = p - δ[c]
    end

    for c in 1:d
        s = scales[c, i] + log(c == l ? factor_long : short_factor)
        scales[c, i] = s
        child_scales[c, j] = s
    end

    o = inverse_sigmoid(clamp(
        opacity_factor * NU.sigmoid(opacities[1, i]),
        1f-6, 1f0 - 1f-6))
    opacities[1, i] = o
    child_opacities[1, j] = o
end

"""
Drop the `recovery_prune_fraction` of Gaussians with the lowest opacity.

Run a few hundred iterations after an opacity reset, this separates the
Gaussians that earned their opacity back — the ones the reconstruction actually
needs — from those that were only propping up an overfit.

Prunes an exact count (rather than thresholding on the quantile *value*) because
right after a reset a large share of opacities are pinned to the same number,
and a value threshold would take all of them at once.
"""
function recovery_prune!(strategy::ImprovedGSStrategy, gs::GaussianModel, optimizers)
    n = length(gs)
    n_prune = floor(Int, strategy.recovery_prune_fraction * n)
    (n_prune > 0 && n_prune < n) || return 0

    o = Array(reshape(NU.sigmoid.(gs.opacities), :))
    valid = ones(Bool, n)
    valid[partialsortperm(o, 1:n_prune)] .= false

    valid_mask = adapt(get_backend(gs), valid)
    prune_points!(strategy, gs, optimizers, valid_mask)
    KA.unsafe_free!(valid_mask)
    return n_prune
end

# ---------------------------------------------------------------------------
# Edge-aware score
# ---------------------------------------------------------------------------

# Canny's 5×5 Gaussian & 3×3 Sobel, transcribed from the reference implementation.
# The Sobel pair is one kernel and its transpose.
const CANNY_GAUSSIAN_5x5 = SMatrix{5, 5, Float32}(
    2, 4, 5, 4, 2,
    4, 9, 12, 9, 4,
    5, 12, 15, 12, 5,
    4, 9, 12, 9, 4,
    2, 4, 5, 4, 2) ./ 159f0

const CANNY_SOBEL_3x3 = SMatrix{3, 3, Float32}(
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1)

"""
Per-Gaussian edge-aware score, or `nothing` when there is nothing to score with
(no dataset/rasterizer, or an empty view sample) — the caller then falls back to
a uniform draw over the gradient candidates.

`edge_sample_cams` train views are re-rendered forward-only with their edge map
as per-pixel weights; each view's scores are normalized to `[0, 1]` before being
averaged into the total, so a low-contrast view still gets an equal vote.
Gaussians the view did not touch keep their running value.

Views are drawn without replacement (see [`sample_score_views!`](@ref)); their
images are re-read from disk on background tasks, so the reads overlap the renders.
"""
function edge_scores(
    strategy::ImprovedGSStrategy, gs::GaussianModel, rast, dataset;
    step::Int, view_target = nothing,
)
    (rast ≡ nothing || dataset ≡ nothing) && return nothing
    ids = sample_score_views!(strategy, length(dataset.train_cameras), step)
    isempty(ids) && return nothing

    kab = get_backend(gs)
    n = length(gs)
    width, height = view_resolution(dataset)
    scores = KA.zeros(kab, Float32, n)
    view_scores = KA.zeros(kab, Float32, n)
    view_weight = 1f0 / length(ids)
    # Without a `view_target` nobody can say what the loss supervises here, so
    # the whole image counts (`post_train_step!` called outside a `Trainer`).
    load = view_target ≡ nothing ?
        (id -> (load_image(dataset, id, :train), nothing, nothing)) :
        view_target

    # Reads run ahead of the renders, but only a few at a time: a full scan is
    # the whole train set & they do not all fit in host memory at once.
    pending = [Threads.@spawn(load(ids[i]))
        for i in 1:min(VIEW_LOOKAHEAD, length(ids))]
    for (k, id) in enumerate(ids)
        image, mask, sky_mask = fetch(popfirst!(pending))::Tuple{
            Array{UInt8, 3}, Maybe{Matrix{Float32}}, Maybe{Matrix{Float32}}}
        ahead = k + VIEW_LOOKAHEAD
        ahead ≤ length(ids) && push!(pending, Threads.@spawn(load(ids[ahead])))

        edge_map!(strategy, device_image(kab, image), id;
            weight=supervision_weight(kab, mask, sky_mask, width, height))

        fill!(view_scores, 0f0)
        rast(
            gs.points, gs.opacities, gs.scales, gs.rotations,
            gs.features_dc, gs.features_rest;
            camera=dataset.train_cameras[id], sh_degree=gs.sh_degree,
            edge_scores=view_scores, edge_map=strategy.edge_map)

        lo, hi = minimum(view_scores), maximum(view_scores)
        hi > lo || continue
        _accumulate_edge_scores!(kab, 256)(
            scores, view_scores, rast.gstate.radii,
            lo, view_weight / (hi - lo); ndrange=n)
    end
    KA.unsafe_free!(view_scores)
    return scores
end

"""
Per-pixel weight of what the loss actually supervises in a view: inside its
coverage mask, outside its sky mask. `nothing` when the view has neither.

A split can only improve a region the loss looks at. Beyond the coverage mask
the target is blacked out, and in the sky region the alpha loss is emptying the
Gaussians rather than refining them — edges there must not attract splits.
"""
function supervision_weight(kab, mask, sky_mask, width::Int, height::Int)
    w = mask ≡ nothing ? nothing : device_map(kab, mask, width, height)
    sky_mask ≡ nothing && return w
    sky = 1f0 .- device_map(kab, sky_mask, width, height)
    return w ≡ nothing ? sky : w .* sky
end

@kernel cpu=false inbounds=true function _accumulate_edge_scores!(
    scores::AbstractVector{Float32},
    @Const(view_scores), @Const(radii),
    lo::Float32, scale::Float32,
)
    i = @index(Global)
    radii[i] > 0i32 || return
    scores[i] += (view_scores[i] - lo) * scale
end

"""
Take up to `edge_sample_cams` view ids from the pool, refilling it with a fresh
shuffle when it runs dry, so a sweep covers every view before repeating any.
`edge_full_scan_iters` steps take the whole train set instead.
"""
function sample_score_views!(strategy::ImprovedGSStrategy, n_views::Int, step::Int)
    n_views > 0 || return Int[]
    (step in strategy.edge_full_scan_iters || strategy.edge_sample_cams ≥ n_views) &&
        return collect(1:n_views)

    ids = Int[]
    while length(ids) < strategy.edge_sample_cams
        isempty(strategy.view_pool) && (strategy.view_pool = shuffle(1:n_views))
        push!(ids, pop!(strategy.view_pool))
    end
    return ids
end

"""
Canny edge detection through non-maximum suppression, on the luminance of
`target_image` (`(3, width, height)`), normalized by the median of its
positive values so scores are comparable across views.

There is no hysteresis/double-threshold stage and no binarization: the result is
the continuous NMS-surviving gradient magnitude, which is what makes it usable
as a per-pixel *weight* rather than a mask.

`weight` scales the result before it is normalized, to keep the score inside
what the loss supervises (see [`supervision_weight`](@ref)).
"""
function edge_map!(
    strategy::ImprovedGSStrategy, target_image, view_id::Int;
    weight::Maybe{AbstractMatrix{Float32}} = nothing,
)
    kab = get_backend(strategy.edge_map)
    # `(c, w, h)`, as `device_image` hands it over.
    width, height = size(target_image, 2), size(target_image, 3)

    if size(strategy.edge_map) != (width, height)
        KA.unsafe_free!(strategy.edge_map)
        KA.unsafe_free!(strategy.edge_blur)
        KA.unsafe_free!(strategy.edge_grad)
        # `@uncached`: these outlive the refine & must not be recycled from under us.
        strategy.edge_map = GPUArrays.@uncached KA.zeros(kab, Float32, (width, height))
        strategy.edge_blur = GPUArrays.@uncached KA.zeros(kab, Float32, (width, height))
        strategy.edge_grad = GPUArrays.@uncached KA.zeros(kab, SVector{2, Float32}, (width, height))
        # Medians are resolution-dependent: a resize invalidates the cache.
        empty!(strategy.edge_medians)
    end

    _canny_blur!(kab)(strategy.edge_blur, target_image; ndrange=(width, height))
    _canny_sobel!(kab)(strategy.edge_grad, strategy.edge_blur; ndrange=(width, height))
    _canny_nms!(kab)(strategy.edge_map, strategy.edge_grad, strategy.edge_min; ndrange=(width, height))
    # Before the median, so it reflects the supervised region only.
    # `weight` is fixed per view, so the cache stays valid either way.
    weight ≡ nothing || (strategy.edge_map .*= weight)

    median = get!(strategy.edge_medians, view_id) do
        positive = filter(>(0f0), vec(Array(strategy.edge_map)))
        isempty(positive) ? 1f0 : partialsort!(positive, cld(length(positive), 2))
    end
    strategy.edge_map ./= max(median, 1f-9)
    return strategy.edge_map
end

@kernel cpu=false inbounds=true function _canny_blur!(
    blur::AbstractMatrix{Float32}, @Const(image),
)
    x, y = @index(Global, NTuple)
    w, h = size(blur, 1), size(blur, 2)

    total = 0f0
    for dy in -2:2, dx in -2:2
        xi = clamp(x + dx, 1, w)
        yi = clamp(y + dy, 1, h)
        luma =
            0.299f0 * image[1, xi, yi] +
            0.587f0 * image[2, xi, yi] +
            0.114f0 * image[3, xi, yi]
        total += CANNY_GAUSSIAN_5x5[dy + 3, dx + 3] * luma
    end
    blur[x, y] = total
end

@kernel cpu=false inbounds=true function _canny_sobel!(
    grad::AbstractMatrix{SVector{2, Float32}}, @Const(blur),
)
    x, y = @index(Global, NTuple)
    w, h = size(blur, 1), size(blur, 2)

    gx, gy = 0f0, 0f0
    # TODO unroll
    for dy in -1:1, dx in -1:1
        xi = clamp(x + dx, 1, w)
        yi = clamp(y + dy, 1, h)
        v = blur[xi, yi]
        # The kernel & its transpose: `gx` varies along `dx`, `gy` along `dy`.
        gx += CANNY_SOBEL_3x3[dy + 2, dx + 2] * v
        gy += CANNY_SOBEL_3x3[dx + 2, dy + 2] * v
    end
    grad[x, y] = SVector{2, Float32}(gx, gy)
end

"""
Round a unit-vector component to the nearest of `{-1, 0, 1}`: the neighbour
offset the gradient direction points at. Written branchlessly rather than as
`round(Int32, x)`, which compiles in an `InexactError` path (and the GPU-side
allocation that goes with it) that can never be taken here.
"""
@inline _nms_step(u::Float32) = ifelse(abs(u) < 0.5f0, 0i32, ifelse(u > 0f0, 1i32, -1i32))

@kernel cpu=false inbounds=true function _canny_nms!(
    edges::AbstractMatrix{Float32}, @Const(grad), floor_mag::Float32,
)
    x, y = @index(Global, NTuple)
    w, h = size(edges, 1), size(edges, 2)

    g = grad[x, y]
    mag = hypot(g[1], g[2])

    # Convolving a *constant* region does not cancel exactly in floating point
    # — `gy` accumulates to `-4v` before the positive taps come back — so flat
    # areas carry a roundoff floor of a few ULP. Non-maximum suppression cannot
    # remove it (a plateau is its own maximum), and since flat pixels vastly
    # outnumber edge pixels that noise would become the normalizing median and
    # scale the real edges into irrelevance. Drop it here: `floor_mag` sits far
    # below the response of even a 1/255 step.
    mag = ifelse(mag < floor_mag, 0f0, mag)

    if mag > 0f0
        # Suppress this pixel unless it is the local maximum along the
        # (rounded) gradient direction.
        dx = _nms_step(g[1] / mag)
        dy = _nms_step(g[2] / mag)

        fwd = grad[clamp(x + dx, 1, w), clamp(y + dy, 1, h)]
        bwd = grad[clamp(x - dx, 1, w), clamp(y - dy, 1, h)]
        if mag < hypot(fwd[1], fwd[2]) || mag < hypot(bwd[1], bwd[2])
            mag = 0f0
        end
    end
    edges[x, y] = mag
end
