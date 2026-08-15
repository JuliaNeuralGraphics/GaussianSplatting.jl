# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
Base.@kwdef struct OptimizationParams
    λ_dssim::Float32 = 0.2f0

    lr_points_start::Float32 = 16f-5
    lr_points_end::Float32 = 16f-7
    lr_points_steps::Int = 30_000

    lr_feature::Float32 = 25f-4
    lr_opacities::Float32 = 5f-2
    lr_scales::Float32 = 5f-3
    lr_rotations::Float32 = 1f-3

    # Sparse Adam (see `sparse_adam.jl`):
    # skips the moment-buffer decay and parameter write for gaussians
    # not visible in the current view's rasterization (per `rast.gstate.radii`),
    # instead of updating all `N` gaussians densely every step.
    use_sparse_adam::Bool = false

    # Composite each train render over a random background instead of the black one.
    # Over black, `c·α + bkgd·T` cannot see `T` at all:
    # a half-transparent gaussian with a brighter color renders exactly
    # like an opaque one, so nothing in the photometric loss asks a surface to become opaque.
    # A background that changes every step does.
    #
    # The reference implementation keeps it off & the published numbers are
    # without it, but with coverage masks it is the recommended setting: the
    # target is composited over the same color (see `masking.jl`), which turns
    # the mask into a claim about occupancy that the photometric term can state
    # at full weight rather than one `mask_opacity_weight` has to carry alone.
    random_background::Bool = false

    # Depth supervision with monocular priors (see `depth_supervision.jl`).
    # Requires depth maps next to the dataset images, an init point cloud and a `:rgbd` rasterizer.
    use_depth_loss::Bool = false
    depth_loss_weight::Float32 = 2f0
    depth_loss_mode::Symbol = :ssi # :ssi (auto), :ssi_disparity, :ssi_depth
    depth_loss_steps::Int = 30_000 # Weight decays to `depth_loss_final_scale` by this step.
    # What the depth weight decays *to*, as a fraction of itself: depth
    # dominates early geometry formation & the photometric loss has to win fine
    # detail late, but the constraint is not dropped entirely (see `step!`).
    depth_loss_final_scale::Float32 = 0.02f0
    # Weight of the depth loss' gradient-matching term relative to its data
    # term: aligns depth *edges* rather than absolute values, so raising it
    # buys sharper discontinuities at the cost of following the prior's own
    # (see `ssi_depth_loss`).
    depth_loss_gradient_weight::Float32 = 1f0

    # Sky dome (see `sky_dome.jl`): a frozen far-field shell of Gaussians
    # composited behind the scene, so sky pixels have a parallax-free place to
    # live instead of becoming near opaque floaters.
    use_sky_dome::Bool = false
    # `:hemisphere` covers only the sky, leaving black behind downward-looking
    # rays so the ground has to become opaque on its own; a full `:sphere`
    # gives the optimizer a free background everywhere & tends to absorb ground
    # into the dome (see `sky_dome.jl`).
    sky_dome_shape::Symbol = :hemisphere # :hemisphere | :sphere
    sky_dome_points::Int = 32_768
    sky_dome_radius_factor::Float32 = 100f0 # × the dataset's camera extent.
    sky_dome_lr::Float32 = 25f-4            # Colors only; matches `lr_feature`.

    # Sky mask supervision (see `sky_dome.jl`): pulls the scene's accumulated
    # alpha to zero on sky rays, so a floater there costs something.
    # Inert unless sky masks were found next to the dataset images.
    use_sky_loss::Bool = false
    sky_loss_weight::Float32 = 1f0
    sky_loss_from_iter::Int = 500

    # Coverage masks (see `masking.jl`): a `masks/` directory next to the
    # dataset images, applied to the targets so the subject trains against the render background.
    # Inert unless masks were found, hence on by default.
    # Pair with `random_background`, which is what makes them geometric.
    #
    # Gates the loss side. `ColmapDataset` takes a `use_masks` of its own for
    # the load side (the carve & the dropped views); the GUI & `main` pass this
    # one through to it, so the two agree.
    use_masks::Bool = false
    # Pull of the accumulated alpha toward zero outside the mask.
    mask_opacity_weight::Float32 = 0.1f0
    mask_opacity_from_iter::Int = 500

    # Bilateral grid appearance modeling (see `bilateral_grid.jl`):
    # per-train-image low-res affine color grids applied to the render before
    # the photometric loss, absorbing exposure / white-balance drift.
    use_bilateral_grid::Bool = false
    bilateral_grid_size::NTuple{3, Int} = (16, 16, 8) # (x, y, guidance)
    bilateral_grid_lr::Float32 = 2f-3
    bilateral_grid_lr_steps::Int = 30_000 # LR decays to 1% by this step.
    tv_loss_weight::Float32 = 10f0

    # Geometry regularization (see `geometry_regularization.jl`): depth-normal
    # consistency + flattening along the thinnest axis.
    # Requires a `:rgbdn` rasterizer, which renders the extra normal channels.
    use_normal_loss::Bool = false
    normal_consistency_weight::Float32 = 0.05f0
    normal_flatten_weight::Float32 = 0.005f0
    # Both terms start once the geometry is roughly in place.
    normal_from_iter::Int = 10_000
end

function lr_exp_scheduler(lr_start::Float32, lr_end::Float32, steps::Int)
    function _scheduler(step::Int)
        (step < 0 || (lr_start ≈ 0f0 && lr_end ≈ 0f0)) && return 0f0

        t = clamp(Float32(step / steps), 0f0, 1f0)
        return exp(log(lr_start) * (1 - t) + log(lr_end) * t)
    end
    return _scheduler
end

# Not a `KA.unsafe_free!` method — `NU.Adam` is not ours to extend.
function free_optimizer!(opt::NU.Adam)
    foreach(KA.unsafe_free!, opt.μ)
    foreach(KA.unsafe_free!, opt.ν)
    return
end

# E.g. `ROCBackend()` -> "ROCBackend".
backend_name(kab) = string(nameof(typeof(kab)))

"""
Device memory (in bytes) held by a scene component.

This is what the app itself allocates, not what the driver reports: the
backend's memory pool keeps freed blocks around, so the process always holds
at least this much and usually more.
"""
memory_usage(x::AbstractArray) = sizeof(x)
memory_usage(::Nothing) = 0
memory_usage(opt::NU.Adam) =
    sum(memory_usage, opt.μ; init=0) + sum(memory_usage, opt.ν; init=0)

mse(x, y) = mean((x .- y).^2)

# Split from `psnr` for the callers that already have the MSE — a masked one,
# say (see `masking.jl`) — and must not recompute it unmasked.
psnr_from_mse(v::Real) = 20f0 * log10(1f0 / sqrt(v))

psnr(x, y) = psnr_from_mse(mse(x, y))

"""
Round a render to the 8-bit sRGB grid the ground truth lives on.

The reference implementation scores PNGs written to disk, so its published
numbers include this rounding; targets here are already 8-bit, only the render
is continuous.
"""
quantize8(x) = floor.(clamp.(x, 0f0, 1f0) .* 255f0 .+ 0.5f0) .* (1f0 / 255f0)

within_gradient(x) = false
CRC.rrule(::typeof(within_gradient), x) = true, _ -> (NoTangent(), NoTangent())

"""
Bilinear upsample of a `(width, height)` map on the device.

Matches `ImageTransformations.imresize`'s convention exactly: outer corners of
the two grids are mapped onto each other, so the source coordinate of output
pixel `i` is `(i - 0.5) * (n_in / n_out) + 0.5`, clamped to the source (which is
what `imresize` does for an upscale, having no extrapolation to fall back on).

Only upscales. A downscale needs `imresize`'s antialiasing, and its output is
small enough that the host does it cheaply — see [`device_map`](@ref).
"""
@kernel cpu=false inbounds=true function _upsample_bilinear!(
    dst::AbstractMatrix{Float32}, @Const(src),
    scale::SVector{2, Float32}, src_size::SVector{2, Int32},
)
    i, j = @index(Global, NTuple)

    x = clamp(scale[1] * (Float32(i) - 0.5f0) + 0.5f0, 1f0, Float32(src_size[1]))
    y = clamp(scale[2] * (Float32(j) - 0.5f0) + 0.5f0, 1f0, Float32(src_size[2]))

    x₀ = unsafe_trunc(Int32, x) # `x ≥ 1`, so truncation is a floor.
    y₀ = unsafe_trunc(Int32, y)
    x₁ = min(x₀ + 1i32, src_size[1])
    y₁ = min(y₀ + 1i32, src_size[2])
    fx = x - Float32(x₀)
    fy = y - Float32(y₀)

    top = src[x₀, y₀] + (src[x₁, y₀] - src[x₀, y₀]) * fx
    bottom = src[x₀, y₁] + (src[x₁, y₁] - src[x₀, y₁]) * fx
    dst[i, j] = top + (bottom - top) * fy
end

function upsample_bilinear(src::AbstractMatrix{Float32}, width::Int, height::Int)
    kab = get_backend(src)
    dst = KA.allocate(kab, Float32, (width, height))
    _upsample_bilinear!(kab)(
        dst, src,
        SVector{2, Float32}(size(src, 1) / width, size(src, 2) / height),
        SVector{2, Int32}(size(src, 1), size(src, 2));
        ndrange=(width, height))
    return dst
end

"""
A per-view map (coverage mask, sky mask, depth prior) on the device at the
training resolution.

These files are typically a fraction of the image's size, and expanding them on
the host is what the loader used to spend most of its allocation on: a
full-resolution `Float32` copy per map per view — ~49 MiB each at full
resolution — pushed across the bus every step. Sending the file's own pixels
instead and expanding them here moves ~98 MB/step off the bus & out of the GC's
way (`load_mask` & `load_depth_prior` leave the map small for exactly this).

A map already at the training resolution is uploaded as-is: that is the
downscale case, which the host resampled because `imresize` antialiases it and
the result is small anyway.
"""
device_map(kab, map::AbstractMatrix{Float32}, width::Int, height::Int) =
    size(map) == (width, height) ?
        adapt(kab, map) :
        upsample_bilinear(adapt(kab, map), width, height)
