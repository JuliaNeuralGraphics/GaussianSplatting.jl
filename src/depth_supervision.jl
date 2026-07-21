# Depth supervision: scale-and-shift-invariant loss with monocular depth
# priors (Depth Anything, MiDaS, or any relative depth/disparity maps).
#
# The affine alignment between a prior and the scene is NOT re-fitted
# against the render every iteration. Instead a fixed per-camera *anchor*
# is fitted once at startup against the SfM point cloud, which keeps the
# supervision target absolute and multi-view consistent instead of
# letting the model drag the target along with its own errors.

# Loss.
const DEPTH_LOSS_MIN_ALPHA = 1f-3
const DEPTH_LOSS_RESIDUAL_SCALE = 2f0
const DEPTH_LOSS_GRADIENT_WEIGHT = 1f0
const DEPTH_LOSS_FINAL_SCALE = 0.02f0

"""
Load a depth prior as a `(width, height)` Float32 map, resized to the
training resolution. Also return the quantization step of the source
encoding (1/255 for 8-bit, 1/65535 for 16-bit, 0 for float formats):
it sizes the loss deadband so the model is not pulled onto the prior's
quantization staircase.
"""
function load_depth_prior(path::String, width::Int, height::Int)
    raw = load(path)
    T = eltype(channelview(raw))
    qstep = T <: AbstractFloat ? 0f0 : Float32(eps(T))

    depth = Float32.(Gray.(raw))
    depth = imresize(depth, (height, width))
    return permutedims(depth, (2, 1)), qstep
end

# Anchors: per-camera affine alignment, fitted once at startup.

"""
Affine alignment of a relative depth prior to the scene:
`a·t + b` maps the prior value `t` to inverse depth `1/(z + floor)`
when `disparity` is set, to depth `z` otherwise.
"""
struct DepthAnchor
    a::Float32
    b::Float32
    floor::Float32
    disparity::Float32
end

struct AnchorFit
    a::Float32
    b::Float32
    corr::Float32
    inlier_fraction::Float32
    usable::Bool
end

"""
Least-squares affine fit `y ≈ a·t + b` over paired samples `ts`, `ys`,
returning `(a, b)`. `var_ridge` regularizes the slope: it shrinks toward
zero when the prior's variance approaches the quantization noise floor,
so a near-constant prior yields a flat (uninformative) fit instead of an
arbitrary steep one.
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
RANSAC affine regression `y ≈ a·t + b`, designed to survive the heavily
contaminated sparse SfM clouds that break least-squares + trimming:
LS init for the residual scale (from Median Absolute Deviation),
2-point hypotheses scored by inlier count on a subset,
then two LS refits on the consensus set.
"""
function ransac_affine_fit(
    ts::Vector{Float32}, ys::Vector{Float32};
    ransac_iterations::Int = 256,
    min_anchor_samples::Int = 256,
    anchor_min_inlier_fraction::Float32 = 0.3f0,
    anchor_min_corr::Float32 = 0.35f0,
    score_subset::Int = 16_384,
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

    usable =
        n ≥ min_anchor_samples &&
        inlier_fraction ≥ anchor_min_inlier_fraction &&
        abs(corr) ≥ anchor_min_corr
    return AnchorFit(a, b, corr, inlier_fraction, usable)
end

function robust_aabb(points::Matrix{Float32}; q::Float32 = 0.01f0, pad::Float32 = 0.1f0)
    lo = SVector{3, Float32}(ntuple(i -> quantile(@view(points[i, :]), q), 3))
    hi = SVector{3, Float32}(ntuple(i -> quantile(@view(points[i, :]), 1f0 - q), 3))
    margin = pad .* (hi .- lo)
    return lo .- margin, hi .+ margin
end

"""
Given depth `prior` for a `camera`, project `points` onto image plane
and collect depth values at those pixels, rejecting invalid projections.

Return `(prior depth, point depth)` pairs, where `point depth` is the
depth of the point in camera space after `R * x + t` transformation.
"""
function collect_anchor_samples(
    points::Matrix{Float32}, camera::Camera, prior::Matrix{Float32};
    aabb_min::SVector{3, Float32}, aabb_max::SVector{3, Float32},
    near_plane::Float32 = 0.2f0,
    max_anchor_samples::Int = 262_144,
)
    n = size(points, 2)
    stride = max(1, cld(n, max_anchor_samples))

    (; width, height) = resolution(camera)
    fx, fy = camera.intrinsics.focal
    cx = camera.intrinsics.principal[1] * width
    cy = camera.intrinsics.principal[2] * height
    R = SMatrix{3, 3, Float32}(camera.w2c[1:3, 1:3])
    t = SVector{3, Float32}(camera.w2c[1:3, 4])

    ts, zs = Float32[], Float32[]
    for i in 1:stride:n
        x = SVector{3, Float32}(points[1, i], points[2, i], points[3, i])
        all(aabb_min .≤ x .≤ aabb_max) || continue

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
"""
function fit_depth_anchors(
    points::Matrix{Float32}, cameras::Vector{Camera},
    priors::Vector{Maybe{Matrix{Float32}}};
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

    aabb_min, aabb_max = robust_aabb(points)
    for i in 1:n_cameras
        prior = priors[i]
        prior ≡ nothing && continue

        ts, zs = collect_anchor_samples(points, cameras[i], prior; aabb_min, aabb_max)
        length(ts) < min_anchor_samples && continue
        # A constant prior has no geometry signal.
        var(ts) < flat_prior_var && continue

        depth_floor = max(1f-8, depth_floor_fraction * median(zs))
        fits[i] = (;
            floor=depth_floor,
            disparity=ransac_affine_fit(ts, 1f0 ./ (zs .+ depth_floor); min_anchor_samples),
            depth=ransac_affine_fit(ts, zs; min_anchor_samples))
    end

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
        anchors[i] = DepthAnchor(f.a, f.b, fits[i].floor, Float32(disparity))
        n_anchored += 1
    end

    @info string(
        "Depth supervision: $n_anchored / $n_cameras cameras anchored ",
        "(", disparity ? "disparity" : "depth", " model).")
    return anchors
end

# Fingerprint of exactly the inputs that change the fit: mode, the point
# cloud, and per-camera identity + pose + intrinsics. A mismatch triggers
# a refit (re-run SfM, regenerated depths, new mode, ...). The per-camera
# contribution is combined commutatively because `train_cameras` is
# re-permuted every dataset load.
function depth_anchors_fingerprint(
    points::Matrix{Float32}, cameras::Vector{Camera}, mode::Symbol,
)
    h = hash(mode)
    h = hash(size(points), h)
    h = hash(points, h)

    cam_hash = zero(UInt)
    for cam in cameras
        ch = hash(cam.img_name)
        ch = hash(cam.w2c, ch)
        ch = hash(cam.intrinsics.focal, ch)
        ch = hash(cam.intrinsics.principal, ch)
        ch = hash(cam.intrinsics.resolution, ch)
        cam_hash += ch # Order-independent.
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
    priors::Vector{Maybe{Matrix{Float32}}};
    mode::Symbol = :ssi,
)
    fingerprint = depth_anchors_fingerprint(points, cameras, mode)
    cache_path = joinpath(dirname(depths_dir), "$(basename(depths_dir))_anchors.bson")

    if isfile(cache_path)
        try
            cached = BSON.load(cache_path)
            if cached[:fingerprint] == fingerprint
                by_name = cached[:anchors]::Dict
                @info "Loaded cached depth anchors from `$cache_path`."
                return Maybe{DepthAnchor}[get(by_name, cam.img_name, nothing) for cam in cameras]
            end

            @warn "Depth anchor cache is stale `$cache_path`, recomputing..."
        catch err
            @warn "Failed to load anchor cache from `$cache_path`, recomputing..."
        end
    end

    anchors = fit_depth_anchors(points, cameras, priors; mode)

    by_name = Dict{String, DepthAnchor}()
    for (cam, a) in zip(cameras, anchors)
        a ≡ nothing || (by_name[cam.img_name] = a)
    end
    BSON.bson(cache_path, Dict(:fingerprint => fingerprint, :anchors => by_name))
    @info "Saved depth anchors to `$cache_path`."
    return anchors
end

# The loss.

geman_mcclure(x) = 0.5f0 * x^2 / (1f0 + x^2)

# Zero loss & gradient inside the quantization corridor: without this,
# the robust loss's sign-like gradients drag smooth surfaces onto the
# prior's 8-bit staircase, producing visible terracing.
deadband(r, half) = sign(r) * max(abs(r) - half, 0f0)

"""
Build the per-pixel supervision target from a prior and its anchor:
inverse-depth target `d`, quantization deadband half-width and validity.
For the depth model the half-step is propagated through the inversion
as `half·d²`.
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
    return target, half_band, valid
end

"""
Scale-and-shift-invariant depth loss on the rendered blended depth `D`.

The rendered value is the alpha-normalized expected depth `e = D / α`
mapped to softened inverse depth `p = 1/(e + floor)`; `α = 1 - T` is
treated as a constant (the backward has no alpha-map input), so
gradients flow to the Gaussians only through the depth map.
Data term: alpha-weighted Geman-McClure penalty on the deadbanded
residual, scaled by the alpha-weighted std of `p` (detached).
Gradient term: same penalty on the mismatch of forward-difference
gradients (MiDaS-style), aligning depth edges rather than absolute
values. The sum is normalized by the total alpha.
"""
function ssi_depth_loss(
    depth_img::AbstractMatrix{Float32};
    transmittance::AbstractMatrix{Float32},
    target::AbstractMatrix{Float32},
    half_band::AbstractMatrix{Float32},
    valid::AbstractMatrix{Bool},
    depth_floor::Float32,
    λ_grad::Float32 = DEPTH_LOSS_GRADIENT_WEIGHT,
)
    # TODO(depth-alpha-grad): propagate gradients through α (transmittance).
    # LichtFeld feeds an analytic `grad_alpha = -g·e/α` (the quotient-rule
    # term of `e = D/α`) into its backward, so depth loss shapes Gaussian
    # opacity directly. Our `∇rasterize`/`∇project!` has no adjoint input for
    # the accumulated-alpha map, so there is nowhere to backprop that
    # cotangent — hence `α` is detached here and opacity is only affected
    # indirectly through the depth channel. Extending the rasterizer backward
    # to accept a `vaccum_α` cotangent (mirroring the color/depth channels)
    # would recover LichtFeld's direct opacity supervision.
    α = ignore_derivatives(clamp.(1f0 .- transmittance, 0f0, 1f0))
    w = ignore_derivatives(ifelse.(valid .& (α .> DEPTH_LOSS_MIN_ALPHA), α, 0f0))
    Σα = ignore_derivatives(max(sum(α), 1f0))

    p = 1f0 ./ (depth_img ./ max.(α, 1f-6) .+ depth_floor)

    σ = ignore_derivatives() do
        Σw = max(sum(w), 1f-6)
        μ = sum(w .* p) / Σw
        max(sqrt(max(sum(w .* (p .- μ).^2) / Σw, 0f0)), 1f-6)
    end
    iscale = 1f0 / (DEPTH_LOSS_RESIDUAL_SCALE * σ)

    data = sum(w .* geman_mcclure.(deadband.(p .- target, half_band) .* iscale))

    # Forward differences along x (width) and y (height); pairs are
    # weighted by the lesser alpha and both pixels must be valid.
    hx = (p[2:end, :] .- p[1:(end - 1), :]) .-
        (target[2:end, :] .- target[1:(end - 1), :])
    bx = half_band[2:end, :] .+ half_band[1:(end - 1), :]
    wx = min.(w[2:end, :], w[1:(end - 1), :])
    grad_x = sum(wx .* geman_mcclure.(deadband.(hx, bx) .* iscale))

    hy = (p[:, 2:end] .- p[:, 1:(end - 1)]) .-
        (target[:, 2:end] .- target[:, 1:(end - 1)])
    by = half_band[:, 2:end] .+ half_band[:, 1:(end - 1)]
    wy = min.(w[:, 2:end], w[:, 1:(end - 1)])
    grad_y = sum(wy .* geman_mcclure.(deadband.(hy, by) .* iscale))

    return (data + λ_grad * (grad_x + grad_y)) / Σα
end
