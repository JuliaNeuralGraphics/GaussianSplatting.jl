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

    # Composite each train render over a random background instead of the
    # black one used at evaluation. Helps opacities escape the background
    # color, but the reference implementation keeps it off & the published
    # numbers are without it.
    random_background::Bool = false

    # Depth supervision with monocular priors (see `depth_supervision.jl`).
    # Requires depth maps next to the dataset images, an init point cloud and a `:rgbd` rasterizer.
    use_depth_loss::Bool = true
    depth_loss_weight::Float32 = 2f0
    depth_loss_mode::Symbol = :ssi # :ssi (auto), :ssi_disparity, :ssi_depth
    depth_loss_steps::Int = 30_000 # Weight decays to 2% by this step.

    # Sky dome (see `sky_dome.jl`): a frozen far-field shell of Gaussians
    # composited behind the scene, so sky pixels have a parallax-free place to
    # live instead of becoming near opaque floaters.
    use_sky_dome::Bool = false
    sky_dome_points::Int = 32_768
    sky_dome_radius_factor::Float32 = 100f0 # × the dataset's camera extent.
    sky_dome_lr::Float32 = 25f-4            # Colors only; matches `lr_feature`.

    # Sky mask supervision (see `sky_dome.jl`): pulls the scene's accumulated
    # alpha to zero on sky rays, so a floater there costs something.
    # Inert unless sky masks were found next to the dataset images.
    use_sky_loss::Bool = true
    sky_loss_weight::Float32 = 1f0
    sky_loss_from_iter::Int = 500

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
    normal_flatten_weight::Float32 = 1f0
    # Both terms start once the geometry is roughly in place
    # (LichtFeld's 20% start fraction of a 30k run).
    normal_from_iter::Int = 6_000
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

psnr(x, y) = 20f0 * log10(1f0 / sqrt(mse(x, y)))

"""
Round a render to the 8-bit sRGB grid the ground truth lives on.

The reference implementation scores PNGs written to disk, so its published
numbers include this rounding; targets here are already 8-bit, only the render
is continuous.
"""
quantize8(x) = floor.(clamp.(x, 0f0, 1f0) .* 255f0 .+ 0.5f0) .* (1f0 / 255f0)

within_gradient(x) = false
CRC.rrule(::typeof(within_gradient), x) = true, _ -> (NoTangent(), NoTangent())
