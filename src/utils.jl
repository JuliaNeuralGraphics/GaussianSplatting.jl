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

    dense_percent::Float32 = 1f-2

    densify_from_iter::Int = 500
    densify_until_iter::Int = 15000
    densification_interval::Int = 100
    densify_grad_threshold::Float32 = 2f-4

    opacity_reset_interval::Int = 30_000
end

function lr_exp_scheduler(lr_start::Float32, lr_end::Float32, steps::Int)
    function _scheduler(step::Int)
        (step < 0 || (lr_start ≈ 0f0 && lr_end ≈ 0f0)) && return 0f0

        t = clamp(Float32(step / steps), 0f0, 1f0)
        return exp(log(lr_start) * (1 - t) + log(lr_end) * t)
    end
    return _scheduler
end
