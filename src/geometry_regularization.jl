"""
Geometry regularization constrains shape of the surface
where depth supervision only constrains its location.

- [`depth_normal_consistency_loss`](@ref) pins surface orientation: normals
  derived from the rendered depth map must agree with the alpha-blended
  per-Gaussian normals (see `gaussian_normal` in `rasterization/projection.jl`,
  rendered as channels 6:8 by a `:rgbdn` rasterizer).
- [`flatten_loss`](@ref) pins surface flatness: L1 on the smallest scale axis,
  which is what makes that min-axis normal well-defined in the first place.

Both are training-time only & are enabled together by `OptimizationParams.use_normal_loss` (see `step!`).

!!! warning "Interaction with `MCMCStrategy`"
    [`flatten_loss`](@ref) is a *global, unweighted* prior: it presses on a
    Gaussian carrying irreplaceable structure exactly as hard as on a redundant
    one, and nothing in it is conditioned on visibility or contribution. Only
    the photometric loss pushes back, so regions with weak photometric
    signal — an object's base where it meets the ground, no silhouette edge,
    low contrast, grazing view angles — have nothing to resist it with.

    Under `MCMCStrategy` that asymmetry is amplified into a runaway (see
    `mcmc.jl`): flattened, weakly-supervised Gaussians lose opacity to
    `opacity_reg`, cross into `inject_noise!`'s low-opacity band where they are
    kicked *every step*, and are eventually declared dead and relocated
    elsewhere. The region does not soften, it dissolves — observed as an
    object's base flattening into the ground over a few thousand steps,
    starting at `normal_from_iter`.

    `normal_flatten_weight` therefore defaults to `0.01`, the same order as
    MCMC's own `scale_reg`, rather than the `1.0` that reads naturally from the
    reference. Note it competes with `normal_consistency_weight` (`0.05`),
    which is the term doing the actual geometry work — flattening exists only
    to make the min-axis normal well-defined, so it has no business dominating.
"""

# Thresholds taken from LichtFeld.
const NORMAL_MIN_ALPHA = 0.5f0
const NORMAL_MAX_REL_DEPTH_JUMP = 0.05f0
const NORMAL_MIN_EXPECTED_DEPTH = 1f-6
const NORMAL_MIN_RENDER_NORM = 0.1f0
const NORMAL_MIN_VALID_COUNT = 64f0
const NORMAL_MIN_VALID_WEIGHT = 16f0
const NORMAL_MIN_CROSS_NORM_SQ = 1f-24

"""
Per-axis camera-space ray components for every pixel of `camera`:
`(rx, ry)` such that the ray through the 1-based pixel `(x, y)` is
`(rx[x], ry[y], 1)`.

The half-pixel offset & 1-based convention match `collect_anchor_samples` in `depth_supervision.jl`,
so the depth channel is interpreted the same way by both losses.
"""
function pixel_rays(kab, camera::Camera)
    (; width, height) = resolution(camera)
    fx, fy = camera.intrinsics.focal
    cx = camera.intrinsics.principal[1] * width
    cy = camera.intrinsics.principal[2] * height

    rx = Float32[(Float32(x - 1) + 0.5f0 - cx) / fx for x in 1:width]
    ry = Float32[(Float32(y - 1) + 0.5f0 - cy) / fy for y in 1:height]
    return adapt(kab, rx), adapt(kab, ry)
end

"""
Depth-normal consistency loss on the rendered blended depth `depth`, alpha map `alpha`
and normal map `normals` (a `(3, width, height)` array of camera-space normals),
with `rays` from [`pixel_rays`](@ref).

The alpha-normalized expected depth `e = D / α` is back-projected along the pixel rays;
central differences give two surface tangents whose cross product
is the depth-implied normal `n_d`.
The loss is the alpha-weighted angular disagreement `1 - cos(n_d, n_r)`
against the rendered Gaussian normals.

Both `D` and `α` are differentiable, so the term shapes geometry and opacity;
the weights & the normalizer built from `α` stay detached,
supervision pressure should not leak in through its own weighting.

The `argmin` / flip sign inside `n_d`'s orientation is detached too,
mirroring the rasterizer's own normal flip.

Pixels are used only where the geometry is unambiguous:
the interior of the image, opaque center & 4-neighborhood (`α ≥ 0.5`),
and no relative depth jump above `NORMAL_MAX_REL_DEPTH_JUMP` (not across a silhouette).
A view with too few valid pixels contributes nothing.
"""
function depth_normal_consistency_loss(
    depth::AbstractMatrix{Float32}, alpha::AbstractMatrix{Float32},
    normals::AbstractArray{Float32, 3};
    rays::Tuple{AbstractVector{Float32}, AbstractVector{Float32}},
)
    width, height = size(depth)
    (width > 2 && height > 2) || return 0f0

    rx, ry = rays

    # Interior pixels & their 4-neighborhood, as (w-2, h-2) blocks.
    ix, iy = 2:(width - 1), 2:(height - 1)
    Rx_c = reshape(rx[ix], :, 1)
    Rx_p = reshape(rx[3:width], :, 1)
    Rx_m = reshape(rx[1:(width - 2)], :, 1)
    Ry_c = reshape(ry[iy], 1, :)
    Ry_p = reshape(ry[3:height], 1, :)
    Ry_m = reshape(ry[1:(height - 2)], 1, :)

    # Expected depth `e = D / α`.
    # Every denominator below is clamped, so `cos` stays finite
    # even where the validity mask is zero & `0 · NaN` cannot poison the sum.
    #
    # NOTE: `α` is deliberately not `clamp`ed here.
    # Zygote's `clamp` adjoint is zero *at* the bound,
    # so clamping to `[0, 1]` would silently drop the alpha cotangent on fully opaque pixels,
    # exactly the ones this term trusts most.
    # The lower bound below is all the division needs.
    e = max.(depth, 0f0) ./ max.(alpha, 1f-6)

    e_c = e[ix, iy]
    e_xp, e_xm = e[3:width, iy], e[1:(width - 2), iy]
    e_yp, e_ym = e[ix, 3:height], e[ix, 1:(height - 2)]

    # Surface tangents from central differences of the back-projected points
    # `e · (rx, ry, 1)`.
    δx, δy = e_xp .- e_xm, e_yp .- e_ym
    tx1 = e_xp .* Rx_p .- e_xm .* Rx_m
    tx2 = δx .* Ry_c
    tx3 = δx
    ty1 = δy .* Rx_c
    ty2 = e_yp .* Ry_p .- e_ym .* Ry_m
    ty3 = δy

    n1 = tx2 .* ty3 .- tx3 .* ty2
    n2 = tx3 .* ty1 .- tx1 .* ty3
    n3 = tx1 .* ty2 .- tx2 .* ty1
    n_sq = n1.^2 .+ n2.^2 .+ n3.^2
    n_norm = sqrt.(max.(n_sq, NORMAL_MIN_CROSS_NORM_SQ))

    # Orient toward the camera, like the rasterizer's own flip (detached).
    sign = ignore_derivatives() do
        facing = n1 .* Rx_c .+ n2 .* Ry_c .+ n3
        ifelse.(facing .> 0f0, -1f0, 1f0)
    end
    flip = sign ./ n_norm
    nd1, nd2, nd3 = n1 .* flip, n2 .* flip, n3 .* flip

    nr1 = normals[1, ix, iy]
    nr2 = normals[2, ix, iy]
    nr3 = normals[3, ix, iy]
    nr_sq = nr1.^2 .+ nr2.^2 .+ nr3.^2
    nr_norm = sqrt.(max.(nr_sq, NORMAL_MIN_RENDER_NORM^2))
    cosθ = (nd1 .* nr1 .+ nd2 .* nr2 .+ nd3 .* nr3) ./ nr_norm

    # Validity & weights are constants w.r.t. AD.
    w, count = ignore_derivatives() do
        α = clamp.(alpha, 0f0, 1f0)
        α_c = α[ix, iy]
        opaque =
            (α_c .≥ NORMAL_MIN_ALPHA) .&
            (α[3:width, iy] .≥ NORMAL_MIN_ALPHA) .&
            (α[1:(width - 2), iy] .≥ NORMAL_MIN_ALPHA) .&
            (α[ix, 3:height] .≥ NORMAL_MIN_ALPHA) .&
            (α[ix, 1:(height - 2)] .≥ NORMAL_MIN_ALPHA)
        # A depth jump means the 4-neighborhood straddles a silhouette:
        # the cross product there describes the discontinuity, not a surface.
        jump = NORMAL_MAX_REL_DEPTH_JUMP .* e_c
        continuous =
            (e_c .≥ NORMAL_MIN_EXPECTED_DEPTH) .&
            (abs.(e_xp .- e_c) .≤ jump) .& (abs.(e_xm .- e_c) .≤ jump) .&
            (abs.(e_yp .- e_c) .≤ jump) .& (abs.(e_ym .- e_c) .≤ jump)
        ok =
            opaque .& continuous .&
            isfinite.(e_c) .&
            (n_sq .≥ NORMAL_MIN_CROSS_NORM_SQ) .&
            (nr_sq .≥ NORMAL_MIN_RENDER_NORM^2)
        ifelse.(ok, α_c, 0f0), Float32(sum(ok))
    end

    # Too little evidence in this view:
    # a handful of pixels would make the normalized mean pure noise.
    Σw = ignore_derivatives(sum(w))
    (count ≥ NORMAL_MIN_VALID_COUNT && Σw ≥ NORMAL_MIN_VALID_WEIGHT) || return 0f0

    return sum(w .* (1f0 .- cosθ)) / max(Σw, 1f0)
end

"""
L1 on the smallest scale axis: `mean(exp(min(scales)))` over Gaussians,
flattening each one along its thinnest axis so that its min-axis normal is well-defined.

`scales` are raw (pre-`exp`) and the `argmin` is treated as a constant,
as in LichtFeld's `add_flatten_regularization_grads`, so only the winning axis receives a gradient.

Weighted by `normal_flatten_weight`, which is deliberately small: this term is
scaffolding for [`depth_normal_consistency_loss`](@ref), not a goal in itself,
and it shrinks every Gaussian regardless of whether anything needs it to. See
the interaction warning at the top of this file before raising it.
"""
function flatten_loss(scales::AbstractMatrix{Float32})
    n = size(scales, 2)
    n == 0 && return 0f0

    # One-hot on the argmin, built outside AD.
    # A bare `scales .== minimum(scales; dims=1)` mask would be wrong:
    # `compute_scales` initializes all three axes identically,
    # so ties are the common case early in training & every tied axis would be counted.
    # The `cumsum` keeps only the first tied axis — LichtFeld's tie-break.
    mask = ignore_derivatives() do
        hit = scales .== minimum(scales; dims=1)
        Float32.(hit .& (cumsum(hit; dims=1) .== 1))
    end
    return sum(exp.(scales) .* mask) / n
end
