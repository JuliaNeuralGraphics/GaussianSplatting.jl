# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
Base.@kwdef struct OptimizationParams
    λ_dssim::Float32 = 0.2f0

    lr_points_start::Float32 = 16f-5
    lr_points_end::Float32 = 16f-7
    lr_points_steps::Int = 30_000

    lr_feature::Float32 = 25f-4
    lr_opacities::Float32 = 5f-2
    lr_scales::Float32 = 5f-3
    lr_rotations::Float32 = 1f-3

    dense_percent::Float32 = 1f-2

    densify_from_iter::Int = 500
    densify_until_iter::Int = 15000
    densification_interval::Int = 100
    densify_grad_threshold::Float32 = 2f-4

    opacity_reset_interval::Int = 30_000
end

struct SSIM{W <: Flux.Conv}
    window::W
    c1::Float32
    c2::Float32
end

function SSIM(kab; channels::Int = 3, σ::Float32 = 1.5f0, window_size::Int = 11)
    w = ImageFiltering.KernelFactors.gaussian(σ, window_size)
    w2d = reshape(reshape(w, :, 1) * reshape(w, 1, :), window_size, window_size, 1)
    window = reshape(
        repeat(w2d, 1, 1, channels),
        window_size, window_size, 1, channels)

    conv = Flux.Conv(
        (window_size, window_size), channels => channels;
        pad=(window_size ÷ 2, window_size ÷ 2),
        groups=channels, bias=false)
    copy!(conv.weight, window)

    SSIM(kab != CPU() ? Flux.gpu(conv) : conv, 0.01f0^2, 0.03f0^2)
end

# Inputs are in (W, H, C, B) format.
function (ssim::SSIM)(x::T, ref::T) where T
    μ₁, μ₂ = ssim.window(x), ssim.window(ref)
    μ₁², μ₂² = μ₁.^2, μ₂.^2
    μ₁₂ = μ₁ .* μ₂

    σ₁² = ssim.window(x.^2) .- μ₁²
    σ₂² = ssim.window(ref.^2) .- μ₂²
    σ₁₂ = ssim.window(x .* ref) .- μ₁₂

    l = ((2f0 .* μ₁₂ .+ ssim.c1) .* (2f0 .* σ₁₂ .+ ssim.c2)) ./
        ((μ₁² .+ μ₂² .+ ssim.c1) .* (σ₁² .+ σ₂² .+ ssim.c2))
    return mean(l)
end

function lr_exp_scheduler(lr_start::Float32, lr_end::Float32, steps::Int)
    function _scheduler(step::Int)
        (step < 0 || (lr_start ≈ 0f0 && lr_end ≈ 0f0)) && return 0f0

        t = clamp(Float32(step / steps), 0f0, 1f0)
        return exp(log(lr_start) * (1 - t) + log(lr_end) * t)
    end
    return _scheduler
end
