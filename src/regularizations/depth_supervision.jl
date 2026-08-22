"""
Depth supervision: scale-and-shift-invariant loss with monocular depth priors.

The affine alignment between a prior and the scene is not
re-fitted against the render every iteration.
Instead a fixed per-camera "anchor" is fitted once at startup
against the SfM point cloud, which keeps the supervision target absolute
and multi-view consistent instead of letting the model drag the target
along with its own errors.

An anchor is only fitted where the SfM cloud has points, so it vouches only for
the range of prior values its inliers covered. Outside that bracket the affine
map is extrapolating, and each end is handled on its own terms
(see [`depth_target`](@ref)):

- Past the far end sit the sky pixels, below every prior value the fit ever saw.
  Taken at face value the extrapolation places the sky at a finite depth and
  manufactures floaters, so those pixels are supervised one-sidedly — as a lower
  bound on distance only (see [`ssi_depth_loss`](@ref)).
- Past the near end the extrapolation is simply not used: those pixels are
  dropped from supervision.
"""

const DEPTH_LOSS_MIN_ALPHA = 1f-3
const DEPTH_LOSS_RESIDUAL_SCALE = 2f0

"""
Load a depth prior as a `(width, height)` Float32 map.
Also return the quantization step of the source encoding
(1/255 for 8-bit, 1/65535 for 16-bit, 0 for float formats):
it sizes the loss deadband so the model is not pulled onto the prior's quantization staircase.
"""
function load_depth_prior(path::String, width::Int, height::Int)
    raw = load(path)
    T = eltype(channelview(raw))
    qstep = T <: AbstractFloat ? 0f0 : Float32(eps(T))

    depth = Float32.(Gray.(raw))
    all(size(depth) .≤ (height, width)) || (depth = fit_resolution(depth, (height, width)))
    return permutedims(depth, (2, 1)), qstep
end

# Anchors: per-camera affine alignment, fitted once at startup.

"""
Affine alignment of a relative depth prior to the scene:
`a·t + b` maps the prior value `t` to:
- inverse depth `1 / (z + floor)` when `disparity` is set;
- to depth `z` otherwise.

`p_near` & `p_far` are the largest & smallest target-space (inverse-depth)
values the fit's inlier support covers: the nearest and the farthest distance
the anchor can vouch for. Outside that bracket the affine map is extrapolating,
and the two sides are not equally salvageable — see [`depth_target`](@ref) and
[`ssi_depth_loss`](@ref). `0f0` on either disables that side's check.
"""
struct DepthAnchor
    a::Float32
    b::Float32
    floor::Float32
    disparity::Float32
    p_far::Float32
    p_near::Float32
end

"""
Map a prior value `t` through `anchor` into target space (inverse depth).
The scalar counterpart of the broadcasts in [`depth_target`](@ref).
"""
function anchor_target(a::Float32, b::Float32, floor::Float32, disparity::Float32, t::Float32)
    affine = a * t + b
    return disparity > 0 ?
        min(affine, 1f0 / floor) :
        1f0 / (affine + floor)
end

anchor_target(anchor::DepthAnchor, t::Float32) =
    anchor_target(anchor.a, anchor.b, anchor.floor, anchor.disparity, t)

"""
Build an anchor and derive its support bracket `[p_far, p_near]` from the
prior-value support `[t_lo, t_hi]` the fit was estimated on.

The mapping is monotonic in `t`, so the farther of the two endpoint targets is
simply the smaller one & the nearer is the larger — which resolves the slope
sign & the disparity/depth parameterization without special-casing either.

A bracket has to have width to mean anything: `t_lo == t_hi` says the fit was
supported at a single value, not over a range, and taking that point as the
support boundary would flag everything on one side of one arbitrary target.
Such a bracket, and any non-finite or non-positive bound, yields `0f0` — the
unbracketed, two-sided-everywhere behavior from before this existed.

Not a `DepthAnchor` constructor: with six fields that method would collide with
the struct's own.
"""
function anchor_from_support(
    a::Float32, b::Float32, floor::Float32, disparity::Float32,
    t_lo::Float32, t_hi::Float32,
)
    t_hi > t_lo || return DepthAnchor(a, b, floor, disparity, 0f0, 0f0)

    p_lo = anchor_target(a, b, floor, disparity, t_lo)
    p_hi = anchor_target(a, b, floor, disparity, t_hi)
    p_far, p_near = min(p_lo, p_hi), max(p_lo, p_hi)
    (isfinite(p_far) && p_far > 0f0) || (p_far = 0f0)
    (isfinite(p_near) && p_near > 0f0) || (p_near = 0f0)
    return DepthAnchor(a, b, floor, disparity, p_far, p_near)
end

# Helper structure used by ransac_affine_fit.
# `t_lo`/`t_hi` bracket the prior values of the final inlier set: the range the
# fit is actually supported by, everything outside it being extrapolation.
struct AnchorFit
    a::Float32
    b::Float32
    corr::Float32
    inlier_fraction::Float32
    t_lo::Float32
    t_hi::Float32
    usable::Bool
end

"""
Least-squares affine fit `y ≈ a·t + b` over paired samples `ts`, `ys`, returning `(a, b)`.

`var_ridge` regularizes the slope:
it shrinks toward zero when the prior's variance approaches the quantization noise floor,
so a near-constant prior yields a flat fit instead of an arbitrary steep one.
"""
function ls_affine_fit(ts, ys; var_ridge::Float32 = 1.5f-5)
    μt, μy = mean(ts), mean(ys)
    cov_ty = mean((ts .- μt) .* (ys .- μy))
    var_t = mean(abs2, ts .- μt)
    a = cov_ty / (var_t + var_ridge)
    b = μy - a * μt
    return a, b
end

"""
RANSAC affine regression `y ≈ a·t + b`, designed to survive the residual
outliers of a sparse SfM cloud that break least-squares + trimming:
- LS init for the residual scale (from Median Absolute Deviation);
- 2-point hypotheses scored by inlier count on a subset;
- then two LS refits on the final set.

`anchor_min_inlier_fraction` is a real quality gate rather than a formality
only because [`collect_anchor_samples`](@ref) now feeds this the camera's own
track. Against a cloud projected without a visibility test the honest samples
were a ~12% minority, so any threshold a contaminated fit could clear was one
the correct fit did not need.
"""
function ransac_affine_fit(
    ts::Vector{Float32}, ys::Vector{Float32};
    ransac_iterations::Int = 256,
    min_anchor_samples::Int = 256,
    anchor_min_inlier_fraction::Float32 = 0.6f0,
    anchor_min_corr::Float32 = 0.35f0,
    score_subset::Int = 16_384,
    support_quantile::Float32 = 0.02f0,
)
    n = length(ts)
    a, b = ls_affine_fit(ts, ys)
    res = abs.(ys .- (a .* ts .+ b))
    # Robust inlier threshold ϵ = 3·σ, floored at 1e-8:
    #   median(res) is the MAD (median absolute deviation), robust to the
    #     heavy outlier contamination that would inflate a plain std;
    #   1.4826 = 1/Φ⁻¹(0.75) rescales MAD into a std estimate for Gaussian data;
    #   3 is the σ gate.
    ϵ = max(3f0 * 1.4826f0 * median(res), 1f-8)

    subset = n ≤ score_subset ?
        (1:n) :
        round.(Int, range(1, n; length=score_subset))
    score(a, b) = count(i -> abs(ys[i] - (a * ts[i] + b)) ≤ ϵ, subset)

    best_a, best_b, best_score = a, b, score(a, b)
    for _ in 1:ransac_iterations
        i, j = rand(1:n), rand(1:n)
        δt = ts[i] - ts[j]
        abs(δt) < 1f-8 && continue

        aᵢ = (ys[i] - ys[j]) / δt
        bᵢ = ys[i] - aᵢ * ts[i]
        s = score(aᵢ, bᵢ)
        s > best_score && ((best_a, best_b, best_score) = (aᵢ, bᵢ, s))
    end

    a, b = best_a, best_b
    inliers = Int[]
    for _ in 1:2
        inliers = findall(i -> abs(ys[i] - (a * ts[i] + b)) ≤ ϵ, 1:n)
        length(inliers) < min_anchor_samples && break
        a, b = ls_affine_fit(@view(ts[inliers]), @view(ys[inliers]))
    end

    inlier_fraction = Float32(length(inliers) / n)
    corr = length(inliers) < 2 ?
        0f0 : Float32(cor(@view(ts[inliers]), @view(ys[inliers])))
    isfinite(corr) || (corr = 0f0)

    # Quantiles rather than extrema: a couple of surviving outliers should not
    # stretch the claimed support across the whole prior range.
    t_lo, t_hi = if length(inliers) < 2
        0f0, 0f0
    else
        ti = ts[inliers]
        Float32(quantile(ti, support_quantile)),
        Float32(quantile(ti, 1f0 - support_quantile))
    end

    usable =
        n ≥ min_anchor_samples &&
        inlier_fraction ≥ anchor_min_inlier_fraction &&
        abs(corr) ≥ anchor_min_corr
    return AnchorFit(a, b, corr, inlier_fraction, t_lo, t_hi, usable)
end

"""
Approximate a camera's track with a coarse z-buffer: bucket every projected
point into a `cell`-pixel grid & keep only those within `tolerance` of the
nearest depth in their bucket.

The fallback for datasets whose `images.bin` carries no tracks (RealityCapture
exports, `scripts/realitycapture.jl`). It rejects a point only when some *other*
point lands in the same bucket in front of it, so on a sparse cloud plenty of
occluded points survive — it is a filter, not a visibility oracle. Still far
better than [`collect_anchor_samples`](@ref)'s failure mode, which is to accept
every one of them.
"""
function visible_by_zbuffer(
    points::Matrix{Float32}, camera::Camera;
    cell::Int = 16, tolerance::Float32 = 0.05f0, near_plane::Float32 = 0.2f0,
)
    (; width, height) = resolution(camera)
    fx, fy = camera.intrinsics.focal
    cx = camera.intrinsics.principal[1] * width
    cy = camera.intrinsics.principal[2] * height
    R = SMatrix{3, 3, Float32}(camera.w2c[1:3, 1:3])
    t = SVector{3, Float32}(camera.w2c[1:3, 4])

    nearest = fill(Inf32, cld(width, cell), cld(height, cell))
    # `(index, bucket, depth)` for everything that lands in frame.
    projected = Tuple{UInt32, CartesianIndex{2}, Float32}[]
    for i in 1:size(points, 2)
        p = R * SVector{3, Float32}(points[1, i], points[2, i], points[3, i]) + t
        z = p[3]
        z > near_plane || continue

        px = floor(Int, fx * p[1] / z + cx) + 1
        py = floor(Int, fy * p[2] / z + cy) + 1
        (1 ≤ px ≤ width && 1 ≤ py ≤ height) || continue

        bucket = CartesianIndex(cld(px, cell), cld(py, cell))
        nearest[bucket] = min(nearest[bucket], z)
        push!(projected, (UInt32(i), bucket, z))
    end

    visible = UInt32[]
    for (i, bucket, z) in projected
        z ≤ nearest[bucket] * (1f0 + tolerance) && push!(visible, i)
    end
    return visible
end

"""
Given depth `prior` for a `camera`, project the points `visible` names onto the
image plane and collect depth values at those pixels, rejecting invalid projections.

Return `(prior depth, point depth)` pairs, where `point depth` is the
depth of the point in camera space after `R * x + t` transformation.

!!! warning "Only the camera's own track"
    `visible` must be COLMAP's track for this camera — the points it actually
    saw — not the whole cloud. A depth prior describes the *first* surface along
    each ray, so a point that lands inside the frame from behind an occluder
    pairs that surface's prior value with a depth from behind it. The pairing is
    one-sided (occluded points are always farther, never nearer), so it does not
    average out: it drags the fit apart at both ends, over-predicting near depths
    and under-predicting far ones — a ground plane bowed into a bowl that every
    camera agrees on. On a 237-view orbit of a fountain, projecting the whole
    cloud made ~88% of the samples occluded and biased the anchored target by
    +68% at 1 m and −46% at 43 m; the same fit on tracks alone stays within ±5%
    from 2 m to 22 m.
"""
function collect_anchor_samples(
    points::Matrix{Float32}, camera::Camera, prior::Matrix{Float32},
    visible::Vector{UInt32};
    near_plane::Float32 = 0.2f0,
    max_anchor_samples::Int = 262_144,
)
    n = length(visible)
    stride = max(1, cld(n, max_anchor_samples))

    (; width, height) = resolution(camera)
    fx, fy = camera.intrinsics.focal
    cx = camera.intrinsics.principal[1] * width
    cy = camera.intrinsics.principal[2] * height
    R = SMatrix{3, 3, Float32}(camera.w2c[1:3, 1:3])
    t = SVector{3, Float32}(camera.w2c[1:3, 4])

    ts, zs = Float32[], Float32[]
    for k in 1:stride:n
        i = visible[k]
        x = SVector{3, Float32}(points[1, i], points[2, i], points[3, i])

        p = R * x + t
        z = p[3]
        z > near_plane || continue

        px = floor(Int, fx * p[1] / z + cx) + 1
        py = floor(Int, fy * p[2] / z + cy) + 1
        (1 ≤ px ≤ width && 1 ≤ py ≤ height) || continue

        tp = prior[px, py]
        (isfinite(tp) && tp > 0f0) || continue
        push!(ts, tp)
        push!(zs, z)
    end
    return ts, zs
end

"""
Fit per-camera depth anchors against the SfM point cloud.

Each camera with a prior gets two candidate fits:
- disparity (`1 / (z + floor) ≈ a·t + b`);
- and depth (`z ≈ a·t + b`).

Where `floor` softens the inversion so near-camera outliers cannot dominate.
With `mode = :ssi` the dataset-wide parameterization is resolved by majority vote over per-camera correlations,
while `:ssi_disparity` and `:ssi_depth` force it.
Cameras whose selected fit is unusable or has an inconsistent slope sign are dropped from depth supervision.

`visible_points[i]` names the columns of `points` camera `i` observed (COLMAP's
track). Only those are sampled — see [`collect_anchor_samples`](@ref) for why
the rest of the cloud is not a harmless surplus.

`load_prior(i)` returns camera `i`'s prior or `nothing`: the priors are read
from disk one at a time (`ColmapDataset` does not hold them), and this whole
pass is skipped when the anchor cache is warm — see [`load_or_fit_depth_anchors`](@ref).
"""
function fit_depth_anchors(
    points::Matrix{Float32}, cameras::Vector{Camera},
    visible_points::Vector{Vector{UInt32}},
    load_prior;
    mode::Symbol = :ssi,
    min_anchor_samples::Int = 256,
    depth_floor_fraction::Float32 = 0.05f0,
    flat_prior_var::Float32 = 1f-6,
)
    modes = (:ssi, :ssi_disparity, :ssi_depth)
    mode in modes || error("Invalid depth loss mode: $mode ∉ $modes")

    n_cameras = length(cameras)
    anchors = Vector{Maybe{DepthAnchor}}(nothing, n_cameras)
    fits = Vector{Maybe{NamedTuple}}(nothing, n_cameras)
    # Kept for `report_anchor_bias` once the parameterization vote has settled.
    samples = Vector{Maybe{Tuple{Vector{Float32}, Vector{Float32}}}}(nothing, n_cameras)

    n_without_track = 0
    for i in 1:n_cameras
        # One prior at a time: they are read from disk here (`load_prior`) and
        # dropped again, never all held at once.
        prior = load_prior(i)
        prior ≡ nothing && continue

        visible = visible_points[i]
        if isempty(visible)
            n_without_track += 1
            visible = visible_by_zbuffer(points, cameras[i])
        end

        ts, zs = collect_anchor_samples(points, cameras[i], prior, visible)
        length(ts) < min_anchor_samples && continue
        # A constant prior has no geometry signal.
        var(ts) < flat_prior_var && continue

        depth_floor = max(1f-8, depth_floor_fraction * median(zs))
        samples[i] = (ts, zs)
        fits[i] = (;
            floor=depth_floor,
            disparity=ransac_affine_fit(ts, 1f0 ./ (zs .+ depth_floor); min_anchor_samples),
            depth=ransac_affine_fit(ts, zs; min_anchor_samples))
    end

    n_without_track > 0 && @warn string(
        "$n_without_track / $n_cameras cameras have no SfM track in `images.bin`; ",
        "anchors there fall back to a coarse z-buffer visibility test, which is ",
        "an approximation (see `visible_by_zbuffer`). Reconstruct with COLMAP, ",
        "or expect the anchors on those views to be less accurate.")

    disparity = if mode == :ssi
        votes, total = 0, 0
        for fit in fits
            fit ≡ nothing && continue
            (fit.disparity.usable || fit.depth.usable) || continue
            total += 1
            better_disparity =
                !fit.depth.usable ||
                (fit.disparity.usable && abs(fit.disparity.corr) ≥ abs(fit.depth.corr))
            votes += better_disparity
        end
        votes ≥ total - votes
    else
        mode == :ssi_disparity
    end
    @info "Depth supervision mode: `$(disparity ? :disparity : :depth)`."

    # Majority slope sign among usable fits: outvoted cameras are dropped.
    selected(fit) = disparity ? fit.disparity : fit.depth
    sign_vote = sum(fits) do fit
        fit ≡ nothing && return 0
        f = selected(fit)
        f.usable ? Int(sign(f.a)) : 0
    end
    slope_sign = sign_vote ≥ 0 ? 1f0 : -1f0

    n_anchored = 0
    for i in 1:n_cameras
        fits[i] ≡ nothing && continue
        f = selected(fits[i])
        (f.usable && sign(f.a) == slope_sign) || continue
        anchors[i] = anchor_from_support(
            f.a, f.b, fits[i].floor, Float32(disparity), f.t_lo, f.t_hi)
        n_anchored += 1
    end

    @info string(
        "Depth supervision: $n_anchored / $n_cameras cameras anchored ",
        "(", disparity ? "disparity" : "depth", " model).")
    report_anchor_bias(anchors, samples)
    return anchors
end

# Bins the anchor residuals by true depth, one line per octave.
const ANCHOR_BIAS_BIN_RATIO = 2f0
const ANCHOR_BIAS_MIN_SAMPLES = 64
# Bins holding less than this share of the samples are reported but excluded
# from the warning: an octave with a few hundred points at the edge of the
# reconstruction says little about the geometry the training actually sees, and
# the affine map is always at its worst there.
const ANCHOR_BIAS_CORE_FRACTION = 0.05f0
# A trend this large across the scene's core is the signature of a fit pulled
# apart by samples it should never have seen.
const ANCHOR_BIAS_WARN_SPREAD = 0.15f0

"""
Report how far the fitted anchors place the SfM points they were fitted on,
as a median relative error binned by true depth.

Only samples inside the anchor's support bracket `[p_far, p_near]` are counted,
so the table describes the supervision as it is actually applied: outside the
bracket the near side is unsupervised and the far side is a one-sided bound,
and a "bias" for either would not mean what the column says it means.

An anchor is an affine map, so it cannot follow a prior whose relation to depth
is not affine — the leftover curvature shows up here as a trend across the bins,
near depths pushed out and far ones pulled in (or the reverse).

Read the two columns against each other. `bias` is where the target sits and
`spread` is how much the cameras disagree about it, so a large bias over a small
spread is the dangerous combination: the views agree on a wrong answer and
reinforce it into a warped surface, where disagreement would merely have blurred
one. A flat `bias` column is the healthy case, whatever the spread.
"""
function report_anchor_bias(
    anchors::Vector{Maybe{DepthAnchor}},
    samples::Vector{Maybe{Tuple{Vector{Float32}, Vector{Float32}}}},
)
    # Bin index is the octave of the true depth, shared across all cameras.
    bins = Dict{Int, Vector{Float32}}()
    for (anchor, sample) in zip(anchors, samples)
        (anchor ≡ nothing || sample ≡ nothing) && continue
        ts, zs = sample
        for (t, z) in zip(ts, zs)
            z > 0f0 || continue
            # `anchor_target` is inverse depth either way, so this inverts back
            # to the depth the supervision actually asks for.
            p = anchor_target(anchor, t)
            p > 0f0 || continue
            # Extrapolated samples are not supervised as locations (see
            # `depth_target`), so a bias for them would describe nothing.
            (anchor.p_far > 0f0 && p < anchor.p_far) && continue
            (anchor.p_near > 0f0 && p > anchor.p_near) && continue
            z_pred = 1f0 / p - anchor.floor
            isfinite(z_pred) || continue
            push!(get!(() -> Float32[], bins, floor(Int, log(z) / log(ANCHOR_BIAS_BIN_RATIO))),
                z_pred / z - 1f0)
        end
    end
    isempty(bins) && return

    total = sum(length, values(bins))
    lines = String[]
    core_biases = Float32[]
    for k in sort!(collect(keys(bins)))
        errors = bins[k]
        length(errors) < ANCHOR_BIAS_MIN_SAMPLES && continue
        bias = median(errors)
        # Half the 16..84 percentile range: the 1σ equivalent, but robust.
        spread = 0.5f0 * (quantile(errors, 0.84f0) - quantile(errors, 0.16f0))
        core = length(errors) ≥ ANCHOR_BIAS_CORE_FRACTION * total
        core && push!(core_biases, bias)
        push!(lines, string(
            "  ", rpad(string(round(ANCHOR_BIAS_BIN_RATIO^k; digits=2), "..",
                round(ANCHOR_BIAS_BIN_RATIO^(k + 1); digits=2)), 16),
            lpad(string(round(100 * bias; digits=1), "%"), 8),
            lpad(string(round(100 * spread; digits=1), "%"), 10),
            lpad(string(length(errors)), 10),
            core ? "" : "  (tail)"))
    end
    isempty(lines) && return

    @info string(
        "Depth anchor bias (`z_target / z_sfm - 1` by true depth, ",
        "inside the fitted support):\n",
        "  ", rpad("depth", 16), lpad("bias", 8), lpad("spread", 10),
        lpad("samples", 10), "\n",
        join(lines, "\n"))

    length(core_biases) < 2 && return
    spread = maximum(core_biases) - minimum(core_biases)
    spread > ANCHOR_BIAS_WARN_SPREAD && @warn string(
        "Depth anchors are biased by $(round(100 * spread; digits=1))% across ",
        "the scene's depth range: the affine model does not fit these priors, ",
        "so the supervision target is warped & every view agrees on the warp. ",
        "Expect flat surfaces to bow. Check the priors' parameterization ",
        "(`depth_loss_mode`) before trusting the geometry.")
    return
end

function depth_anchors_fingerprint(
    points::Matrix{Float32}, cameras::Vector{Camera},
    visible_points::Vector{Vector{UInt32}}, mode::Symbol,
)
    h = hash(mode)
    h = hash(size(points), h)
    h = hash(points, h)
    h = hash(visible_points, h)

    cam_hash = zero(UInt)
    for cam in cameras
        ch = hash(cam.img_name)
        ch = hash(cam.w2c, ch)
        ch = hash(cam.intrinsics.focal, ch)
        ch = hash(cam.intrinsics.principal, ch)
        ch = hash(cam.intrinsics.resolution, ch)
        cam_hash += ch # Independent of the camera order.
    end
    return hash(cam_hash, h)
end

"""
Fit per-camera depth anchors, or load them from the cache next to
`depths_dir` when a cache with a matching fingerprint exists.
"""
function load_or_fit_depth_anchors(
    depths_dir::String,
    points::Matrix{Float32}, cameras::Vector{Camera},
    visible_points::Vector{Vector{UInt32}},
    load_prior;
    mode::Symbol = :ssi,
)
    fingerprint = depth_anchors_fingerprint(points, cameras, visible_points, mode)
    cache_path = joinpath(dirname(depths_dir), "$(basename(depths_dir))_anchors.toml")

    if isfile(cache_path)
        try
            cached = TOML.parsefile(cache_path)
            # The hash is a `UInt`: TOML integers are signed, so it is a string.
            if parse(UInt, cached["fingerprint"]) == fingerprint
                by_name = cached["anchors"]::Dict
                @info "Loaded cached depth anchors from `$cache_path`."
                return Maybe{DepthAnchor}[
                    haskey(by_name, cam.img_name) ?
                        DepthAnchor(Float32.(by_name[cam.img_name])...) : nothing
                    for cam in cameras]
            end

            @warn "Depth anchor cache is stale `$cache_path`, recomputing..."
        catch err
            @warn "Failed to load anchor cache from `$cache_path`, recomputing..."
        end
    end

    anchors = fit_depth_anchors(points, cameras, visible_points, load_prior; mode)

    by_name = Dict{String, Any}()
    for (cam, a) in zip(cameras, anchors)
        a ≡ nothing || (by_name[cam.img_name] =
            [to_toml(getfield(a, f)) for f in fieldnames(DepthAnchor)])
    end
    open(cache_path, "w") do io
        println(io, "# GaussianSplatting.jl depth anchor cache.")
        println(io, "# `[a, b, floor, disparity, p_far, p_near]` per image, see `DepthAnchor`.")
        TOML.print(io, Dict{String, Any}(
            "fingerprint" => string(fingerprint),
            "anchors" => by_name); sorted=true)
    end
    @info "Saved depth anchors to `$cache_path`."
    return anchors
end

geman_mcclure(x) = 0.5f0 * x^2 / (1f0 + x^2)

# Zero loss & gradient inside the quantization corridor: without this,
# the robust loss's sign-like gradients drag smooth surfaces onto the
# prior's 8-bit staircase, producing visible terracing.
deadband(r, half) = sign(r) * max(abs(r) - half, 0f0)

"""
Build the per-pixel supervision target from a prior and its anchor:
inverse-depth target `d`, quantization deadband half-width, validity and the
far-extrapolation flag.
For the depth model the half-step is propagated through the inversion
as `half·d²`.

Both ends of the fit's inlier support are enforced, asymmetrically, because the
two extrapolations fail differently:

- `far_extrap` marks pixels beyond the far end (`anchor.p_far`) — typically the
  sky, which no SfM point ever constrained. Kept, and used one-sidedly by
  [`ssi_depth_loss`](@ref): the extrapolated value is trustworthy as a lower
  bound on distance, and that bound is what suppresses sky floaters.
- Pixels nearer than `anchor.p_near` are dropped from `valid` outright. The
  mirror-image bound would read "nothing may sit farther than the nearest thing
  the fit vouches for", which pushes geometry *toward* the camera — onto real
  surfaces, where being wrong costs something, and at the depths where the
  inverse-depth residual has its steepest gradient (`|dp/dz| = 1/z²`). There is
  no floater-like failure it would buy back in exchange, so the extrapolation is
  not used at all rather than used as a bound. It is also a sliver of the frame,
  unlike the block of sky that motivates the far end.

Dropping them via `valid` also removes them from the residual scale & the
gradient term, since both are built from it.
"""
function depth_target(anchor::DepthAnchor, prior::AbstractMatrix{Float32}, qstep::Float32)
    affine = anchor.a .* prior .+ anchor.b
    valid = isfinite.(prior) .& (prior .> 0f0) .& (affine .> 0f0)
    half_step = 0.5f0 * qstep * abs(anchor.a)
    if anchor.disparity > 0
        target = min.(affine, 1f0 / anchor.floor)
        half_band = fill!(similar(prior), half_step)
    else
        target = 1f0 ./ (affine .+ anchor.floor)
        half_band = half_step .* target.^2
    end
    far_extrap = target .< anchor.p_far
    # `p_near == 0` means the fit reported no usable support bracket.
    anchor.p_near > 0f0 && (valid = valid .& (target .≤ anchor.p_near))
    return target, half_band, valid, far_extrap
end

"""
Scale-and-shift-invariant depth loss on the rendered blended depth `D`
and the rendered alpha map `alpha`.

The rendered value is the alpha-normalized expected depth `e = D / α`
mapped to softened inverse depth `p = 1/(e + floor)`.
Both `D` and `α` are differentiable rasterizer outputs,
so the quotient rule feeds an alpha cotangent back into the backward
and the depth loss shapes Gaussian opacity directly.

The weights & normalizations built from `α` stay detached:
supervision pressure should not leak in through its own weighting.

Data term:
alpha-weighted Geman-McClure penalty on the deadbanded residual,
scaled by the alpha-weighted std of `p` (detached).

On `far_extrap` pixels the residual is one-sided: only a render *nearer* than
the target is penalized, never one farther away. Their target is the affine
map evaluated outside the range of prior values it was fitted on (the sky, in
practice), so it is trustworthy as a lower bound on distance and nothing more.
Read as a constraint: nothing may sit closer than the farthest thing the fit
can vouch for — which is exactly what suppresses sky floaters — while the
extrapolated value itself never pulls geometry forward onto it.

The near end of the support needs no counterpart here: [`depth_target`](@ref)
has already cleared those pixels out of `valid`, so they reach neither this term
nor the scale above.

Gradient term:
same penalty on the mismatch of forward-difference gradients,
aligning depth edges rather than absolute values. Extrapolated pixels are
excluded from it altogether: a finite difference across the sky/scene boundary
compares a real depth edge against an invented one.

The sum is normalized by the total alpha.
"""
function ssi_depth_loss(
    depth_img::AbstractMatrix{Float32},
    alpha::AbstractMatrix{Float32};
    target::AbstractMatrix{Float32},
    half_band::AbstractMatrix{Float32},
    valid::AbstractMatrix{Bool},
    far_extrap::AbstractMatrix{Bool},
    depth_floor::Float32,
    λ_grad::Float32, # `OptimizationParams.depth_loss_gradient_weight`.
)
    α = ignore_derivatives(clamp.(alpha, 0f0, 1f0))
    w = ignore_derivatives(ifelse.(valid .& (α .> DEPTH_LOSS_MIN_ALPHA), α, 0f0))
    Σα = ignore_derivatives(max(sum(α), 1f0))
    # `1` where the data term's residual goes one-sided, and `w` restricted to
    # the pixels the fit's support actually covers — i.e. whose target is an
    # interpolation & so means something as a location, not just as a bound.
    one_sided = ignore_derivatives(ifelse.(far_extrap, 1f0, 0f0))
    w_supported = ignore_derivatives(w .* (1f0 .- one_sided))

    # NOTE: the differentiable path uses `alpha`, not the clamped `α`. Zygote's
    # `clamp` adjoint is zero *at* the bound, so a fully opaque pixel would
    # silently lose the alpha cotangent this loss exists to produce.
    p = 1f0 ./ (depth_img ./ max.(alpha, 1f-6) .+ depth_floor)

    # Residual scale from the supported pixels only: a large block of sky at
    # `p ≈ 0` would otherwise inflate it for the whole image.
    #
    # This cuts both ways & the balance is not measured. Excluding the sky also
    # *shrinks* σ, which raises `iscale` and saturates Geman-McClure sooner;
    # since the largest residuals in inverse-depth space belong to near objects
    # (`|dz/dp| = 1/p²`), a tighter scale is felt first by close geometry.
    # Note the effect is largest early in training — once sky alpha collapses,
    # `w` is ≈ 0 there anyway and the two variants converge.
    σ = ignore_derivatives() do
        Σw = max(sum(w_supported), 1f-6)
        μ = sum(w_supported .* p) / Σw
        max(sqrt(max(sum(w_supported .* (p .- μ).^2) / Σw, 0f0)), 1f-6)
    end
    iscale = 1f0 / (DEPTH_LOSS_RESIDUAL_SCALE * σ)

    # `r - min(r, 0)` is `max(r, 0)`, so the mask selects the one-sided
    # residual with plain arithmetic — no `ifelse` in the differentiable path.
    r = deadband.(p .- target, half_band)
    r = r .- one_sided .* min.(r, 0f0)
    data = sum(w .* geman_mcclure.(r .* iscale))

    # Forward differences along x (width) and y (height); pairs are weighted by
    # the lesser alpha and both pixels must be valid & inside the fit's support.
    hx =
        (p[2:end, :] .- p[1:(end - 1), :]) .-
        (target[2:end, :] .- target[1:(end - 1), :])
    bx = half_band[2:end, :] .+ half_band[1:(end - 1), :]
    wx = min.(w_supported[2:end, :], w_supported[1:(end - 1), :])
    grad_x = sum(wx .* geman_mcclure.(deadband.(hx, bx) .* iscale))

    hy =
        (p[:, 2:end] .- p[:, 1:(end - 1)]) .-
        (target[:, 2:end] .- target[:, 1:(end - 1)])
    by = half_band[:, 2:end] .+ half_band[:, 1:(end - 1)]
    wy = min.(w_supported[:, 2:end], w_supported[:, 1:(end - 1)])
    grad_y = sum(wy .* geman_mcclure.(deadband.(hy, by) .* iscale))

    return (data + λ_grad * (grad_x + grad_y)) / Σα
end
