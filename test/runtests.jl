# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

# ENV["GSP_TEST_AMDGPU"] = true
# ENV["GSP_TEST_CUDA"] = true

import Pkg
if get(ENV, "GSP_TEST_AMDGPU", "false") == "true"
    @info "`GSP_TEST_AMDGPU` is `true`, importing AMDGPU.jl."
    Pkg.add("AMDGPU")
    using AMDGPU

    kab = ROCBackend()
elseif get(ENV, "GSP_TEST_CUDA", "false") == "true"
    @info "`GSP_TEST_CUDA` is `true`, importing CUDA.jl."
    Pkg.add(["CUDA", "cuDNN"])
    using CUDA, cuDNN

    kab = CUDABackend()
else
    error("No GPU backend was specified.")
end

using Adapt
using Test
using Zygote
using FiniteDifferences
using LinearAlgebra
using GaussianSplatting
using Statistics
using Random
using StaticArrays
using Quaternions
using Rotations
using Flux
using ImageFiltering

using GaussianSplatting: i32, u32

import KernelAbstractions as KA

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

    SSIM(kab != KA.CPU() ? Flux.gpu(conv) : conv, 0.01f0^2, 0.03f0^2)
end

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

DATASET = nothing
GAUSSIANS = nothing

# @info "Testing on `$kab` backend."

@testset "GaussianSplatting" begin

@testset "quat2mat" begin
    r = RotXYZ(rand(Float32), rand(Float32), rand(Float32))
    q = QuatRotation{Float32}(r)

    ŷ = @inferred GaussianSplatting.unnorm_quat2rot(SVector{4, Float32}(q.w, q.x, q.y, q.z))
    y = SMatrix{3, 3, Float32, 9}(q)
    @test all(ŷ .≈ y)
end

@testset "∇unnorm_quat2rot vs finite differences" begin
    for _ in 1:100
        # Un-normalized quaternions with norms well below & above 1:
        # the adjoint must handle the normalization, not assume ‖q‖ = 1.
        scale = 0.3f0 + 2f0 * rand(Float32)
        q = SVector{4, Float32}(randn(Float32, 4)...) * scale
        vR = SMatrix{3, 3, Float32, 9}(randn(Float32, 9)...)

        vq = @inferred GaussianSplatting.∇unnorm_quat2rot(q, vR)

        # Differentiate the scalar loss L(q) = Σᵢⱼ vR[i, j]·R(q)[i, j] in
        # Float64 (converting to Float32 only for the primal call), so the
        # finite-difference error stays well below the tolerance.
        loss = x -> sum(
            SMatrix{3, 3, Float64, 9}(vR) .*
            GaussianSplatting.unnorm_quat2rot(SVector{4, Float32}(x))
        )

        # The loss goes through the Float32 kernel, so its evaluations carry
        # ~1e-6 noise, not the ~1e-16 Float64 roundoff the step-size heuristic
        # assumes: `factor` scales the assumed roundoff so the FDM picks a
        # larger step & doesn't amplify that noise into the gradient.
        # `max_range` keeps that step clear of the singularity at `q = 0`.
        fdm = central_fdm(5, 1; factor=1e10, max_range=0.25 * norm(q))
        fd = FiniteDifferences.grad(fdm, loss, Float64.(collect(q)))[1]
        @test vq ≈ SVector{4, Float64}(fd) atol=1e-3 rtol=5e-3

        # R(c·q) = R(q) ⇒ the gradient has no radial component.
        @test abs(vq ⋅ q) / norm(vq) < 1f-5
    end
end

@testset "∇pos_world_to_cam vs finite differences" begin
    fdm = central_fdm(5, 1; factor=1e10)
    for _ in 1:100
        R = SMatrix{3, 3, Float32, 9}(randn(Float32, 9)...)
        t = SVector{3, Float32}(randn(Float32, 3)...)
        p = SVector{3, Float32}(randn(Float32, 3)...)
        v = SVector{3, Float32}(randn(Float32, 3)...)

        vR, vt, vp = @inferred GaussianSplatting.∇pos_world_to_cam(R, t, p, v)

        loss = (R̂, t̂, p̂) -> SVector{3, Float64}(v) ⋅ GaussianSplatting.pos_world_to_cam(
            SMatrix{3, 3, Float32, 9}(R̂),
            SVector{3, Float32}(t̂),
            SVector{3, Float32}(p̂),
        )
        fd_R, fd_t, fd_p = FiniteDifferences.grad(fdm, loss,
            Matrix{Float64}(R), Vector{Float64}(t), Vector{Float64}(p))
        @test vR ≈ SMatrix{3, 3, Float64, 9}(fd_R) atol=1e-3 rtol=5e-3
        @test vt ≈ SVector{3, Float64}(fd_t) atol=1e-3 rtol=5e-3
        @test vp ≈ SVector{3, Float64}(fd_p) atol=1e-3 rtol=5e-3
    end
end

@testset "∇covar_world_to_cam vs finite differences" begin
    fdm = central_fdm(5, 1; factor=1e10)
    for _ in 1:100
        R = SMatrix{3, 3, Float32, 9}(randn(Float32, 9)...)
        A = SMatrix{3, 3, Float32, 9}(randn(Float32, 9)...)
        Σ = A * A' # Symmetric PSD, like a real covariance.
        vΣ_cam = SMatrix{3, 3, Float32, 9}(randn(Float32, 9)...)
        # The adjoint accumulates on top of an incoming `vR`.
        vR_in = SMatrix{3, 3, Float32, 9}(randn(Float32, 9)...)

        vR, vΣ = @inferred GaussianSplatting.∇covar_world_to_cam(R, Σ, vΣ_cam, vR_in)

        loss = (R̂, Σ̂) -> sum(
            SMatrix{3, 3, Float64, 9}(vΣ_cam) .*
            GaussianSplatting.covar_world_to_cam(
                SMatrix{3, 3, Float32, 9}(R̂),
                SMatrix{3, 3, Float32, 9}(Σ̂)),
        )
        fd_R, fd_Σ = FiniteDifferences.grad(fdm, loss,
            Matrix{Float64}(R), Matrix{Float64}(Σ))
        @test vR - vR_in ≈ SMatrix{3, 3, Float64, 9}(fd_R) atol=1e-3 rtol=5e-3
        @test vΣ ≈ SMatrix{3, 3, Float64, 9}(fd_Σ) atol=1e-3 rtol=5e-3
    end
end

@testset "∇perspective_projection vs finite differences" begin
    focal = SVector{2, Float32}(1000f0, 1000f0)
    resolution = SVector{2, Int32}(1920, 1080)
    principal = SVector{2, Float32}(0.5f0, 0.5f0)

    # FOV clamping limits from the primal (symmetric principal point,
    # so the negative limit equals the positive one): means are placed
    # either well inside or well outside them, and `max_range` bounds
    # the FDM step, so evaluations never cross the clamping kink.
    tan_fov = 0.5f0 .* resolution ./ focal
    lim = (resolution .- principal .* resolution) ./ focal .+ 0.3f0 .* tan_fov
    fdm = central_fdm(5, 1; factor=1e10, max_range=0.1)

    for inside in (true, false), _ in 1:50
        # `x/z` either ≤ 50% of the limit or 20%+ beyond it, random sign.
        ratio = inside ?
            (2f0 .* rand(SVector{2, Float32}) .- 1f0) .* 0.5f0 .* lim :
            sign.(randn(SVector{2, Float32})) .* (1.2f0 .+ 0.5f0 .* rand(SVector{2, Float32})) .* lim
        z = 2f0 + 4f0 * rand(Float32)
        mean = SVector{3, Float32}(ratio[1] * z, ratio[2] * z, z)

        A = 0.1f0 * SMatrix{3, 3, Float32, 9}(randn(Float32, 9)...)
        Σ = A * A' # Symmetric PSD, like a real covariance.
        vΣ_2D = SMatrix{2, 2, Float32, 4}(randn(Float32, 4)...)
        vmean_2D = SVector{2, Float32}(randn(Float32, 2)...)

        vΣ, vmean = @inferred GaussianSplatting.∇perspective_projection(
            mean, Σ, focal, resolution, principal, vΣ_2D, vmean_2D)

        loss = (m̂, Σ̂) -> begin
            Σ_2D, mean_2D = GaussianSplatting.perspective_projection(
                SVector{3, Float32}(m̂), SMatrix{3, 3, Float32, 9}(Σ̂),
                focal, resolution, principal)
            sum(SMatrix{2, 2, Float64, 4}(vΣ_2D) .* Σ_2D) +
                SVector{2, Float64}(vmean_2D) ⋅ mean_2D
        end
        fd_mean, fd_Σ = FiniteDifferences.grad(fdm, loss,
            Vector{Float64}(mean), Matrix{Float64}(Σ))
        @test vmean ≈ SVector{3, Float64}(fd_mean) atol=1e-3 rtol=5e-3
        @test vΣ ≈ SMatrix{3, 3, Float64, 9}(fd_Σ) atol=1e-3 rtol=5e-3
    end
end

@testset "∇quat_scale_to_cov vs finite differences" begin
    for _ in 1:100
        norm_scale = 0.3f0 + 2f0 * rand(Float32)
        q = SVector{4, Float32}(randn(Float32, 4)...) * norm_scale
        scale = exp.(0.5f0 .* SVector{3, Float32}(randn(Float32, 3)...))
        R = GaussianSplatting.unnorm_quat2rot(q)
        vΣ = SMatrix{3, 3, Float32, 9}(randn(Float32, 9)...)

        vq, vscale = @inferred GaussianSplatting.∇quat_scale_to_cov(q, scale, R, vΣ)

        loss = (q̂, ŝ) -> sum(
            SMatrix{3, 3, Float64, 9}(vΣ) .*
            GaussianSplatting.quat_scale_to_cov(
                SVector{4, Float32}(q̂), SVector{3, Float32}(ŝ)),
        )
        fdm = central_fdm(5, 1; factor=1e10, max_range=0.25 * norm(q))
        fd_q, fd_s = FiniteDifferences.grad(fdm, loss,
            Vector{Float64}(q), Vector{Float64}(scale))
        @test vq ≈ SVector{4, Float64}(fd_q) atol=1e-3 rtol=5e-3
        @test vscale ≈ SVector{3, Float64}(fd_s) atol=1e-3 rtol=5e-3
    end
end

@testset "∇inverse vs finite differences" begin
    # `∇inverse` takes the already-inverted matrix, as at its call site.
    # `inverse` & the adjoint assume a symmetric input, so the FD runs over
    # the 3 free entries of `[a b; b c]`: `dL/db` then corresponds to
    # `vX[1, 2] + vX[2, 1]`.
    fdm = central_fdm(5, 1; factor=1e10, max_range=0.2)
    for _ in 1:100
        A = SMatrix{2, 2, Float32, 4}(randn(Float32, 4)...)
        # Positive definite & away from the singular `det ≈ 0` early-out.
        X = A * A' + SMatrix{2, 2, Float32, 4}(0.5f0, 0f0, 0f0, 0.5f0)
        b = randn(Float32, 3)
        vY = SMatrix{2, 2, Float32, 4}(b[1], b[2], b[2], b[3])

        _, Y = GaussianSplatting.inverse(X)
        vX = @inferred GaussianSplatting.∇inverse(Y, vY)

        loss = p -> begin
            _, Ŷ = GaussianSplatting.inverse(
                SMatrix{2, 2, Float32, 4}(p[1], p[2], p[2], p[3]))
            sum(SMatrix{2, 2, Float64, 4}(vY) .* Ŷ)
        end
        fd = FiniteDifferences.grad(fdm, loss, Float64[X[1, 1], X[2, 1], X[2, 2]])[1]
        vX_sym = SVector{3, Float32}(vX[1, 1], vX[1, 2] + vX[2, 1], vX[2, 2])
        @test vX_sym ≈ SVector{3, Float64}(fd) atol=1e-3 rtol=5e-3
    end
end

@testset "∇add_blur vs finite differences" begin
    # `∇add_blur` covers only the `compensation` output (the `Σ_2D_blur`
    # path is an identity, accumulated separately at the call site) & its
    # third argument is the conic — the inverse of the blurred covariance —
    # exactly what the call site passes. Symmetric 3-entry parametrization,
    # as for `∇inverse`.
    ϵ = 0.3f0
    fdm = central_fdm(5, 1; factor=1e10, max_range=0.2)
    for _ in 1:100
        A = SMatrix{2, 2, Float32, 4}(randn(Float32, 4)...)
        Σ = A * A' + SMatrix{2, 2, Float32, 4}(0.5f0, 0f0, 0f0, 0.5f0)
        vcomp = randn(Float32)

        Σ_blur, _, comp = GaussianSplatting.add_blur(Σ, ϵ)
        _, conic = GaussianSplatting.inverse(Σ_blur)
        vΣ = @inferred GaussianSplatting.∇add_blur(comp, vcomp, conic, ϵ)

        loss = p -> Float64(vcomp) * GaussianSplatting.add_blur(
            SMatrix{2, 2, Float32, 4}(p[1], p[2], p[2], p[3]), ϵ)[3]
        fd = FiniteDifferences.grad(fdm, loss, Float64[Σ[1, 1], Σ[2, 1], Σ[2, 2]])[1]
        vΣ_sym = SVector{3, Float32}(vΣ[1, 1], vΣ[1, 2] + vΣ[2, 1], vΣ[2, 2])
        @test vΣ_sym ≈ SVector{3, Float64}(fd) atol=1e-4 rtol=5e-3
    end
end

@testset "∇normalize vs finite differences" begin
    for _ in 1:100
        scale = 0.3f0 + 2f0 * rand(Float32)
        dir = SVector{3, Float32}(randn(Float32, 3)...) * scale
        vdir = SVector{3, Float32}(randn(Float32, 3)...)

        vd = @inferred GaussianSplatting.∇normalize(dir, vdir)

        loss = d̂ -> SVector{3, Float64}(vdir) ⋅ normalize(SVector{3, Float32}(d̂))
        fdm = central_fdm(5, 1; factor=1e10, max_range=0.25 * norm(dir))
        fd = FiniteDifferences.grad(fdm, loss, Vector{Float64}(dir))[1]
        @test vd ≈ SVector{3, Float64}(fd) atol=1e-3 rtol=5e-3
    end
end

@testset "get_rect" begin
    width, height = 1024, 1024
    block = SVector{2, Int32}(16, 16)
    grid = SVector{2, Int32}(cld(width, block[1]), cld(height, block[2]))

    # rect covering only one block
    rmin, rmax = @inferred GaussianSplatting.get_rect(
        SVector{2, Float32}(0, 0), 1i32, grid, block)
    @test all(rmin .== (0, 0))
    @test all(rmax .== (1, 1))

    # rect covering 2 blocks
    rmin, rmax = @inferred GaussianSplatting.get_rect(
        SVector{2, Float32}(0, 0), Int32(block[1] + 1), grid, block)
    @test all(rmin .== (0, 0))
    @test all(rmax .== (2, 2))
end

@testset "ls_affine_fit" begin
    ls_affine_fit = GaussianSplatting.ls_affine_fit

    # Exact affine data is recovered (ridge negligible against real variance).
    ts = collect(Float32, 1:100)
    a, b = ls_affine_fit(ts, 2f0 .* ts .+ 3f0)
    @test a ≈ 2f0 atol=1f-3
    @test b ≈ 3f0 atol=1f-3

    # A constant prior has zero variance: the ridge shrinks the slope to ~0
    # and the intercept falls back to the mean of `ys`.
    a, b = ls_affine_fit(fill(5f0, 100), fill(7f0, 100))
    @test a ≈ 0f0 atol=1f-4
    @test b ≈ 7f0 atol=1f-4
end

@testset "ransac_affine_fit" begin
    ransac_affine_fit = GaussianSplatting.ransac_affine_fit

    # Clean linear data: exact recovery, perfect correlation, all inliers.
    ts = collect(Float32, 1:1000)
    f = ransac_affine_fit(ts, 2f0 .* ts .+ 3f0)
    @test f.a ≈ 2f0 atol=1f-3
    @test f.b ≈ 3f0 atol=1f-3
    @test f.corr ≈ 1f0 atol=1f-3
    @test f.inlier_fraction ≈ 1f0 atol=1f-3
    @test f.usable

    # 25% gross outliers: RANSAC still recovers the slope and stays usable,
    # where a plain least-squares fit would be dragged off the true line.
    Random.seed!(0)
    ys = 2f0 .* ts .+ 3f0
    ys[1:4:end] .= rand(Float32, length(1:4:1000)) .* 3000f0 .- 1000f0
    f = ransac_affine_fit(ts, ys)
    @test f.a ≈ 2f0 atol=1f-1
    @test f.corr > 0.8f0
    @test f.inlier_fraction > 0.6f0
    @test f.usable

    # Pure noise has no linear signal: rejected via the correlation gate.
    Random.seed!(1)
    f = ransac_affine_fit(ts, rand(Float32, 1000))
    @test abs(f.corr) < 0.35f0
    @test !f.usable

    # Too few samples are never usable, regardless of fit quality.
    ts_small = collect(Float32, 1:100)
    f = ransac_affine_fit(ts_small, 2f0 .* ts_small .+ 3f0)
    @test !f.usable

    # The support bracket is reported so `DepthAnchor` can tell interpolation
    # from extrapolation. Quantiles, so a stray inlier cannot stretch it.
    f = ransac_affine_fit(ts, 2f0 .* ts .+ 3f0)
    @test f.t_lo ≈ quantile(ts, 0.02) atol=1f0
    @test f.t_hi ≈ quantile(ts, 0.98) atol=1f0
end

@testset "Depth anchor extrapolation" begin
    DepthAnchor = GaussianSplatting.DepthAnchor
    anchor_target = GaussianSplatting.anchor_target
    depth_target = GaussianSplatting.depth_target

    # Disparity anchor fitted on priors t ∈ [0.3, 0.9]; sky sits at t ≈ 0.
    a, b, dfloor, disparity = 1f0, 0.05f0, 0.1f0, 1f0
    anchor = DepthAnchor(a, b, dfloor, disparity, 0.3f0, 0.9f0)

    # `p_far` is the *smaller* endpoint target, i.e. the farthest distance the
    # fit vouches for, whichever way the slope runs.
    @test anchor.p_far ≈ anchor_target(anchor, 0.3f0)
    @test anchor.p_far < anchor_target(anchor, 0.9f0)
    # A negative slope flips which endpoint is far; `p_far` must follow.
    flipped = DepthAnchor(-a, 1f0, dfloor, disparity, 0.3f0, 0.9f0)
    @test flipped.p_far ≈ anchor_target(flipped, 0.9f0)

    # A zero-width bracket carries no support information: fall back to
    # two-sided supervision everywhere rather than flagging everything below
    # one arbitrary target.
    @test DepthAnchor(a, b, dfloor, disparity, 0f0, 0f0).p_far == 0f0
    @test DepthAnchor(a, b, dfloor, disparity, 0.5f0, 0.5f0).p_far == 0f0
    # With `p_far = 0` nothing is ever flagged.
    flat = DepthAnchor(a, b, dfloor, disparity, 0.5f0, 0.5f0)
    @test !any(depth_target(flat, Float32[0.5 0.005], 1f0 / 255f0)[4])

    # Only the sky pixel is flagged; scene pixels inside the support are not.
    prior = Float32[0.5 0.7; 0.8 0.005]
    target, half_band, valid, far_extrap = depth_target(anchor, prior, 1f0 / 255f0)
    @test all(valid)
    @test far_extrap == Bool[0 0; 0 1]
    # The bug this guards: the extrapolated target is a *finite* depth, so
    # taken two-sidedly it plants geometry there.
    @test isfinite(1f0 / target[2, 2] - dfloor)
end

@testset "One-sided sky depth supervision" begin
    ssi_depth_loss = GaussianSplatting.ssi_depth_loss
    anchor = GaussianSplatting.DepthAnchor(1f0, 0.05f0, 0.1f0, 1f0, 0.3f0, 0.9f0)
    dfloor = anchor.floor

    prior = Float32[0.5 0.7; 0.8 0.005]
    target, half_band, valid, far_extrap = GaussianSplatting.depth_target(
        anchor, prior, 1f0 / 255f0)
    alpha = ones(Float32, 2, 2)
    on_target = 1f0 ./ target .- dfloor
    sel = Float32[0 0; 0 1] # Selects the sky pixel without mutating.

    loss(z, fe) = ssi_depth_loss(
        on_target .* (1f0 .- sel) .+ z .* sel, alpha;
        target, half_band, valid, far_extrap=fe, depth_floor=dfloor)

    sky_z = on_target[2, 2]
    grad(z, fe) = Zygote.gradient(x -> loss(x, fe), z)[1]

    # Nearer than the extrapolated target: penalized, and pushed away.
    @test loss(2f0, far_extrap) > 0f0
    @test grad(2f0, far_extrap) < 0f0
    # Farther: free. This is the whole point — the sky may be arbitrarily
    # distant, it just may not come closer than the fit can vouch for.
    @test loss(10f0 * sky_z, far_extrap) ≈ 0f0 atol=1f-8
    @test grad(10f0 * sky_z, far_extrap) == 0f0

    # Without the flag the same pixel is pulled back onto the extrapolation:
    # the behaviour that manufactures sky floaters.
    two_sided = falses(2, 2)
    @test loss(10f0 * sky_z, two_sided) > 0f0
    @test grad(10f0 * sky_z, two_sided) > 0f0
end

@testset "MCMC relocation (Eq. 9)" begin
    strategy = GaussianSplatting.MCMCStrategy()
    relocation_params = GaussianSplatting.relocation_params

    # ratio = 1: splitting into a single copy is an identity.
    new_o, coeff = relocation_params(strategy, 0.9f0, 1)
    @test new_o ≈ 0.9f0 atol=1f-6
    @test coeff ≈ 1f0 atol=1f-5

    # ratio = 2 closed form: new_o = 1 - √(1 - o), scales shrink.
    o = 0.9f0
    new_o, coeff = relocation_params(strategy, o, 2)
    @test new_o ≈ 1f0 - sqrt(1f0 - o) atol=1f-5
    @test 0f0 < coeff < 1f0

    # Opacity & scale multiplier stay valid & monotonically
    # decrease with the split count over the whole ratio range.
    prev_o, prev_coeff = 1f0, 1f0 + 1f-5
    for ratio in 1:strategy.n_max
        new_o, coeff = relocation_params(strategy, 0.99f0, ratio)
        @test strategy.min_opacity ≤ new_o < 1f0
        @test new_o ≤ prev_o
        @test 0f0 < coeff ≤ prev_coeff
        prev_o, prev_coeff = new_o, coeff
    end

    # Near-dead source stays finite & clamped to the opacity floor.
    new_o, coeff = relocation_params(strategy, 0.004f0, 2)
    @test new_o ≈ strategy.min_opacity
    @test isfinite(coeff) && coeff > 0f0
end

@testset "Tile ranges" begin
    gaussian_keys = adapt(kab, UInt64[0 << 32, 0 << 32, 1 << 32, 2 << 32, 3 << 32])
    ranges = KA.allocate(kab, UInt32, 2, 4)
    fill!(ranges, 0u32)

    GaussianSplatting.identify_tile_range!(kab, 256)(
        ranges, gaussian_keys; ndrange=length(gaussian_keys))
    @test Array(ranges) == UInt32[0; 2;; 2; 3;; 3; 4;; 4; 5;;]
end

@testset "SSIM" begin
    ssim = SSIM(kab)

    x = KA.ones(kab, Float32, (16, 16, 3, 1))
    ref = KA.zeros(kab, Float32, (16, 16, 3, 1))
    @test ssim(x, ref) ≈ 0f0 atol=1f-4 rtol=1f-4
    ref = KA.ones(kab, Float32, (16, 16, 3, 1))
    @test ssim(x, ref) ≈ 1f0

    x = zeros(Float32, (16, 16, 3, 1))
    x[1:4, 1:4, :, :] .= 0.25f0
    x[5:8, 1:4, :, :] .= 0.5f0
    x[9:12, 13:16, :, :] .= 0.75f0
    x[13:16, 13:16, :, :] .= 1f0
    @test ssim(adapt(kab, x), ref) ≈ 0.1035 atol=1f-3 rtol=1f-3

    x = adapt(kab, rand(Float32, 128, 128, 3, 2))
    ref = adapt(kab, rand(Float32, 128, 128, 3, 2))
    @test ssim(x, ref) ≈ mean(GaussianSplatting.fused_ssim(x; ref))

    y, ∇ = Zygote.withgradient(x -> ssim(x, ref), x)
    yf, ∇f = Zygote.withgradient(x -> mean(GaussianSplatting.fused_ssim(x; ref)), x)
    @test y ≈ yf
    @test ∇[1] ≈ ∇f[1]
end

@testset "Bilateral grid" begin
    w, h, n_images = 64, 48, 4
    idx = 2 # Current "train view".

    opt_params = GaussianSplatting.OptimizationParams(;
        use_bilateral_grid=true, bilateral_grid_size=(8, 8, 4))
    bgrid = GaussianSplatting.BilateralGrid(kab, n_images, opt_params)
    image = adapt(kab, rand(Float32, 3, w, h))

    # Identity grids leave the image unchanged & have no variation.
    out = GaussianSplatting.bilateral_slice(image, bgrid.grids[:, :, :, :, idx])
    @test Array(out) ≈ Array(image)
    @test GaussianSplatting.tv_loss(bgrid.grids) ≈ 0f0

    # Gradients through slice + TV, as in `step!`.
    target = adapt(kab, rand(Float32, 3, w, h))
    loss, ∇ = Zygote.withgradient(bgrid.grids, image) do grids, img
        corrected = GaussianSplatting.bilateral_slice(img, grids[:, :, :, :, idx])
        mean(abs.(corrected .- target)) + opt_params.tv_loss_weight * GaussianSplatting.tv_loss(grids)
    end
    ∇grids, ∇image = ∇
    @test isfinite(loss)
    @test size(∇grids) == size(bgrid.grids)
    @test size(∇image) == size(image)
    @test isfinite(sum(∇grids))
    @test isfinite(sum(∇image))
    # Identity grids are constant along guidance, so the z-path cancels &
    # only the sliced view receives a photometric gradient.
    other = [i for i in 1:n_images if i != idx]
    @test all(iszero, Array(∇grids[:, :, :, :, other]))
    @test maximum(abs.(Array(∇grids[:, :, :, :, idx]))) > 0f0
end

@testset "gaussian_normal" begin
    for _ in 1:100
        q = SVector{4, Float32}(randn(Float32, 4)...) * (0.3f0 + 2f0 * rand(Float32))
        scale = exp.(0.5f0 .* SVector{3, Float32}(randn(Float32, 3)...))
        R_w2c = SMatrix{3, 3, Float32, 9}(rand(RotMatrix{3, Float32}))
        R_g = GaussianSplatting.unnorm_quat2rot(q)
        mean_cam = SVector{3, Float32}(
            randn(Float32), randn(Float32), 1f0 + 5f0 * rand(Float32))

        n, k, s = @inferred GaussianSplatting.gaussian_normal(
            R_w2c, R_g, scale, mean_cam)

        # A rotation column is already unit-length: the thinnest axis, in
        # camera space, always oriented back toward the camera.
        @test norm(n) ≈ 1f0
        @test n ⋅ mean_cam ≤ 0f0
        @test scale[k] == minimum(scale)
        @test abs(s) == 1f0
        @test n ≈ s .* (R_w2c * R_g[:, k])
    end
end

@testset "∇gaussian_normal vs finite differences" begin
    fdm = central_fdm(5, 1; factor=1e10, max_range=0.05)
    for _ in 1:100
        norm_scale = 0.3f0 + 2f0 * rand(Float32)
        q = SVector{4, Float32}(randn(Float32, 4)...) * norm_scale
        # Well-separated axes: the argmin must not flip under the perturbation.
        scale = exp.(
            SVector{3, Float32}(0f0, 1f0, 2f0) .+
            0.1f0 .* SVector{3, Float32}(randn(Float32, 3)...))
        R_w2c = SMatrix{3, 3, Float32, 9}(rand(RotMatrix{3, Float32}))
        mean_cam = SVector{3, Float32}(
            randn(Float32), randn(Float32), 2f0 + 5f0 * rand(Float32))
        vnormal = SVector{3, Float32}(randn(Float32, 3)...)

        R_g = GaussianSplatting.unnorm_quat2rot(q)
        n, k, s = GaussianSplatting.gaussian_normal(R_w2c, R_g, scale, mean_cam)
        # The flip sign is detached, so it is a step discontinuity for FD:
        # skip configurations sitting on the boundary.
        abs(n ⋅ normalize(mean_cam)) > 0.1f0 || continue

        vR_g = GaussianSplatting.∇gaussian_normal_column(k, s .* (R_w2c' * vnormal))
        vq, vscale = GaussianSplatting.∇quat_scale_to_cov(
            q, scale, R_g, zeros(SMatrix{3, 3, Float32, 9}), vR_g)
        # `argmin` is a constant: the normal path gives no scale gradient.
        @test all(iszero, vscale)

        loss = q̂ -> begin
            R̂ = GaussianSplatting.unnorm_quat2rot(SVector{4, Float32}(q̂))
            n̂, _, _ = GaussianSplatting.gaussian_normal(R_w2c, R̂, scale, mean_cam)
            sum(SVector{3, Float64}(vnormal) .* n̂)
        end
        fd_q = FiniteDifferences.grad(fdm, loss, Vector{Float64}(q))[1]
        @test vq ≈ SVector{4, Float64}(fd_q) atol=1e-3 rtol=5e-3
    end
end

@testset "flatten_loss" begin
    # Column minima: 0, 1, 3.
    scales = adapt(kab, Float32[
        1f0 2f0 3f0;
        0f0 5f0 4f0;
        2f0 1f0 6f0])
    @test GaussianSplatting.flatten_loss(scales) ≈ mean(exp.(Float32[0, 1, 3]))

    _, (∇,) = Zygote.withgradient(GaussianSplatting.flatten_loss, scales)
    g = Array(∇)
    @test count(!iszero, g) == 3 # Only the thinnest axis is pulled.
    @test g[2, 1] ≈ exp(0f0) / 3
    @test g[3, 2] ≈ exp(1f0) / 3
    @test g[1, 3] ≈ exp(3f0) / 3

    # All axes tied is the initialization case (`compute_scales`): exactly one
    # axis per Gaussian must win, otherwise the term is counted three times.
    tied = adapt(kab, ones(Float32, 3, 4))
    @test GaussianSplatting.flatten_loss(tied) ≈ exp(1f0)
    _, (∇tied,) = Zygote.withgradient(GaussianSplatting.flatten_loss, tied)
    @test count(!iszero, Array(∇tied)) == 4

    @test GaussianSplatting.flatten_loss(adapt(kab, zeros(Float32, 3, 0))) == 0f0
end

@testset "Depth-normal consistency loss" begin
    width, height = 64, 48
    camera = GaussianSplatting.Camera(; fx=100f0, fy=100f0, width, height)
    rx, ry = GaussianSplatting.pixel_rays(kab, camera)
    rxh, ryh = Array(rx), Array(ry)
    @test length(rxh) == width && length(ryh) == height

    # A slanted plane `n ⋅ X = d` in camera space, sampled along the pixel
    # rays: `X = e · (rx, ry, 1)` ⇒ `e = d / (n ⋅ ray)`. `n` faces the camera
    # (negative z) & `d < 0` keeps the plane in front of it.
    n = normalize(SVector{3, Float32}(0.2f0, -0.3f0, -1f0))
    d = -5f0
    depth = Float32[
        d / (n[1] * rxh[x] + n[2] * ryh[y] + n[3])
        for x in 1:width, y in 1:height]
    @test all(>(0f0), depth)

    alpha = ones(Float32, width, height)
    plane_normals = zeros(Float32, 3, width, height)
    for c in 1:3
        plane_normals[c, :, :] .= n[c]
    end

    D, A = adapt(kab, depth), adapt(kab, alpha)
    N = adapt(kab, plane_normals)
    loss = c -> GaussianSplatting.depth_normal_consistency_loss(
        D, A, c; rays=(rx, ry))

    # Central differences of points sampled on a plane are chords *in* that
    # plane, so their cross product recovers the plane normal exactly.
    @test abs(loss(N)) < 1f-4

    # A fronto-parallel guess disagrees by exactly the plane's tilt.
    flat = zeros(Float32, 3, width, height)
    flat[3, :, :] .= -1f0
    @test loss(adapt(kab, flat)) ≈ 1f0 - (n ⋅ SVector{3, Float32}(0f0, 0f0, -1f0)) rtol=1e-3

    # Transparent views carry no geometry: below `NORMAL_MIN_ALPHA` the term
    # is gated off entirely.
    @test GaussianSplatting.depth_normal_consistency_loss(
        D, adapt(kab, fill(0.4f0, width, height)), N; rays=(rx, ry)) == 0f0

    # Gradients flow to depth, alpha & the rendered normals.
    _, ∇ = Zygote.withgradient(D, A, adapt(kab, flat)) do d, a, nn
        GaussianSplatting.depth_normal_consistency_loss(d, a, nn; rays=(rx, ry))
    end
    for g in ∇
        @test all(isfinite, Array(g))
        @test maximum(abs, Array(g)) > 0f0
    end

    # `e = D / α` ⇒ the two cotangents must satisfy the quotient rule exactly.
    # This is what a `clamp.(alpha, 0f0, 1f0)` on the differentiable path
    # would break: Zygote's `clamp` adjoint is zero *at* the bound, silently
    # dropping the alpha gradient on exactly the fully opaque pixels this
    # term trusts most.
    @test Array(∇[2]) ≈ -(depth ./ alpha) .* Array(∇[1]) rtol=1f-4
end

@testset "Rasterizer `:rgbdn` normal channel" begin
    width, height = 64, 48
    camera = GaussianSplatting.Camera(; fx=100f0, fy=100f0, width, height)

    # A grid of Gaussians on a fronto-parallel plane at `z = 3`, all with an
    # identity rotation & a thin 3rd axis, so every camera-space normal is
    # `-e₃` and the blended normal map must equal `-alpha`.
    xs = range(-0.6f0, 0.6f0; length=8)
    points = adapt(kab, Float32[
        p[i] for i in 1:3, p in vec([(x, y, 3f0) for x in xs, y in xs])])
    n_points = size(points, 2)
    colors = adapt(kab, rand(Float32, 3, n_points))
    scales = adapt(kab, repeat(Float32[log(0.2f0), log(0.2f0), log(0.01f0)], 1, n_points))

    gaussians = GaussianSplatting.GaussianModel(kab,
        points, colors, scales; max_sh_degree=0, isotropic=false)
    gaussians.opacities .= 5f0 # ≈ 0.993 after the sigmoid.

    rast = GaussianSplatting.GaussianRasterizer(kab, camera; mode=:rgbdn)
    image = rast(
        gaussians.points, gaussians.opacities, gaussians.scales,
        gaussians.rotations, gaussians.features_dc, gaussians.features_rest;
        camera, sh_degree=0)
    @test size(image) == (8, width, height)

    img = Array(image)
    α = img[5, :, :]
    covered = α .> 0.5f0
    @test any(covered)
    @test maximum(abs, img[6, :, :]) < 1f-4 # nx
    @test maximum(abs, img[7, :, :]) < 1f-4 # ny
    @test all(isapprox.(img[8, :, :][covered], -α[covered]; atol=1f-3))

    # The normal channel's cotangent must reach the rotations.
    weights = adapt(kab, randn(Float32, 3, width, height))
    _, (∇rot,) = Zygote.withgradient(gaussians.rotations) do rotations
        features = rast(
            gaussians.points, gaussians.opacities, gaussians.scales,
            rotations, gaussians.features_dc, gaussians.features_rest;
            camera, sh_degree=0)
        sum(features[6:8, :, :] .* weights)
    end
    @test size(∇rot) == size(gaussians.rotations)
    @test all(isfinite, Array(∇rot))
    @test maximum(abs, Array(∇rot)) > 0f0
end

# A partially transparent scene, so `alpha` spans (0, 1) and the composite is
# actually exercised rather than being trivially 0 or 1 everywhere.
function sky_test_scene(kab; opacity::Float32 = 0.5f0)
    xs = range(-0.6f0, 0.6f0; length=6)
    points = adapt(kab, Float32[
        p[i] for i in 1:3, p in vec([(x, y, 3f0) for x in xs, y in xs])])
    n = size(points, 2)
    colors = adapt(kab, rand(Float32, 3, n))
    scales = adapt(kab, fill(log(0.1f0), 3, n))

    gaussians = GaussianSplatting.GaussianModel(kab,
        points, colors, scales; max_sh_degree=0, isotropic=false)
    gaussians.opacities .= GaussianSplatting.inverse_sigmoid(opacity)
    return gaussians
end

@testset "Sky composite identity" begin
    width, height = 64, 48
    camera = GaussianSplatting.Camera(; fx=100f0, fy=100f0, width, height)
    gaussians = sky_test_scene(kab)
    rast = GaussianSplatting.GaussianRasterizer(kab, camera; mode=:rgbd)

    render(background) = Array(rast(
        gaussians.points, gaussians.opacities, gaussians.scales,
        gaussians.rotations, gaussians.features_dc, gaussians.features_rest;
        camera, sh_degree=0, background))

    # Compositing a *uniform* dome outside the rasterizer must reproduce what
    # the kernel does with the same color as its background. This is the claim
    # the whole design rests on: `alpha` (channel 5) is exactly `1 - T_final`,
    # so `image + (1 - alpha)·sky` is real back-to-front blending, not an
    # approximation.
    bg = SVector{3, Float32}(0.2f0, 0.7f0, 0.4f0)
    in_kernel = render(bg)[1:3, :, :]

    zeroed = render(zeros(SVector{3, Float32}))
    alpha = zeroed[5, :, :]
    composited = zeroed[1:3, :, :] .+
        reshape(1f0 .- alpha, 1, width, height) .* Array(bg)

    @test 0f0 < minimum(alpha) && maximum(alpha) < 1f0 # Genuinely partial.
    @test maximum(abs, in_kernel .- composited) < 1f-5

    # `composite_sky` itself, on device, against the same reference.
    sky_rgb = adapt(kab, repeat(Array(bg), 1, width, height))
    device = Array(GaussianSplatting.composite_sky(
        adapt(kab, zeroed[1:3, :, :]), adapt(kab, alpha), sky_rgb))
    @test maximum(abs, in_kernel .- device) < 1f-5
end

@testset "Sky dome" begin
    width, height = 64, 48
    camera = GaussianSplatting.Camera(; fx=100f0, fy=100f0, width, height)
    opt_params = GaussianSplatting.OptimizationParams(;
        use_sky_dome=true, sky_dome_points=8192)

    radius = 50f0
    sky = GaussianSplatting.SkyDome(kab, camera, opt_params;
        center=zeros(SVector{3, Float32}), radius,
        color=SVector{3, Float32}(0.2f0, 0.4f0, 0.9f0))
    @test length(sky) == 8192
    # The dome must outlive its own far plane, or `project!` culls all of it.
    @test sky.rast.far_plane > radius

    # No holes: a gap in the shell shows up as a black spot in the sky.
    probe = GaussianSplatting.GaussianRasterizer(kab, camera;
        mode=:rgbd, far_plane=4f0 * radius)
    gs = sky.gaussians
    image = Array(probe(
        gs.points, gs.opacities, gs.scales, gs.rotations,
        gs.features_dc, gs.features_rest; camera, sh_degree=0))
    @test minimum(image[5, :, :]) > 0.99f0

    # The dome renders the color it was initialized with.
    rgb = GaussianSplatting.render_sky(sky, camera)
    @test size(Array(rgb)) == (3, width, height)
    @test all(isapprox.(mean(Array(rgb); dims=(2, 3)), [0.2f0; 0.4f0; 0.9f0]; atol=1f-2))

    # Colors are trainable; the frozen geometry is not touched by the optimizer.
    weights = adapt(kab, randn(Float32, 3, width, height))
    _, (∇dc,) = Zygote.withgradient(gs.features_dc) do features_dc
        sum(GaussianSplatting.render_sky(sky, camera, features_dc) .* weights)
    end
    @test size(∇dc) == size(gs.features_dc)
    @test all(isfinite, Array(∇dc))
    @test maximum(abs, Array(∇dc)) > 0f0

    # Merging for export: one set, dome last, higher SH bands zero-padded to
    # the scene's degree so external viewers read a constant color.
    scene = sky_test_scene(kab)
    scene.max_sh_degree = 3
    scene.features_rest = adapt(kab, randn(Float32, 3, 15, length(scene)))
    merged = GaussianSplatting.merge_sky(scene, sky)
    @test length(merged) == length(scene) + length(sky)
    @test size(merged.features_rest) == (3, 15, length(merged))
    @test all(iszero, Array(merged.features_rest)[:, :, (length(scene) + 1):end])
    @test Array(merged.points)[:, (length(scene) + 1):end] ≈ Array(gs.points)
end

@testset "sky_opacity_loss" begin
    sky_opacity_loss = GaussianSplatting.sky_opacity_loss

    alpha = adapt(kab, Float32[0.9 0.1; 0.5 1.0])
    mask = adapt(kab, Float32[1 0; 0 1])
    # Masked mean of α², normalized by the mask weight.
    @test sky_opacity_loss(alpha, mask) ≈ (0.9f0^2 + 1f0^2) / 2f0

    _, (∇,) = Zygote.withgradient(a -> sky_opacity_loss(a, mask), alpha)
    g = Array(∇)
    @test all(iszero, g[[2, 3]])       # Unmasked pixels are untouched.
    @test g[1, 1] > 0f0 && g[2, 2] > 0f0 # Masked ones are pushed toward zero.
    # The saturated pixel is exactly the floater this targets: its gradient
    # must survive (the trap `clamp` would spring, see `ssi_depth_loss`).
    @test g[2, 2] ≈ 2f0 * 1f0 / 2f0

    # An empty mask must not divide by zero.
    @test sky_opacity_loss(alpha, adapt(kab, zeros(Float32, 2, 2))) == 0f0
end

@testset "Checkpoint" begin
    # Mirrors what `save_state` writes: nested named tuples of arrays, a
    # `Camera`, plain scalars.
    adam(dims) = (;
        μ=[fill(1.5f0, d) for d in dims],
        ν=[fill(2.5f0, d) for d in dims],
        current_step=7)
    gaussians(n) = (;
        points=rand(Float32, 3, n),
        features_dc=rand(Float32, 3, 1, n),
        features_rest=rand(Float32, 3, 15, n),
        scales=rand(Float32, 3, n),
        rotations=rand(Float32, 4, n),
        opacities=rand(Float32, 1, n),
        sh_degree=2, max_sh_degree=3)

    camera = GaussianSplatting.Camera(; fx=100f0, fy=100f0, width=64, height=48)
    doc = Dict(
        :gaussians => gaussians(100),
        :optimizers => (; points=adam([(3, 100)]), scales=adam([(3, 100)])),
        :bilateral => (;
            grids=rand(Float32, 12, 8, 8, 4, 5),
            opt=adam([(12, 8, 8, 4, 5)])),
        :step => 30_000,
        :camera => camera,
        :ids => Int32[1, 2, 3])

    dir = mktempdir()
    path = joinpath(dir, "state.bson")
    GaussianSplatting.save_checkpoint(path, doc)
    θ = GaussianSplatting.load_checkpoint(path)

    for k in keys(doc[:gaussians])
        @test getproperty(θ[:gaussians], k) == getproperty(doc[:gaussians], k)
    end
    @test θ[:optimizers].points.μ == doc[:optimizers].points.μ
    @test θ[:optimizers].points.ν == doc[:optimizers].points.ν
    @test θ[:optimizers].scales.current_step == 7
    @test θ[:bilateral].grids == doc[:bilateral].grids
    @test θ[:bilateral].opt.μ == doc[:bilateral].opt.μ
    @test θ[:step] == 30_000
    @test θ[:ids] == Int32[1, 2, 3]
    @test θ[:ids] isa Vector{Int32}

    # Structs stay in the document: the static arrays inside them are not
    # payload & must survive as-is.
    @test θ[:camera] isa GaussianSplatting.Camera
    @test θ[:camera].w2c == camera.w2c
    @test θ[:camera].c2w == camera.c2w
    @test θ[:camera].intrinsics.resolution == camera.intrinsics.resolution

    header_bytes(f) = open(f) do io
        seek(io, length(GaussianSplatting.CHECKPOINT_MAGIC))
        filesize(f) - read(io, UInt64)
    end

    # The point of the format: BSON caps a document at 2 GiB (its length is an
    # `Int32`), so the document must not grow with the array data.
    big = joinpath(dir, "big.bson")
    GaussianSplatting.save_checkpoint(big,
        Dict(doc..., :gaussians => gaussians(1000)))
    @test header_bytes(big) == header_bytes(path)
    @test filesize(big) > filesize(path)

    # Checkpoints written before the blob format are plain BSON.
    legacy = joinpath(dir, "legacy.bson")
    GaussianSplatting.BSON.bson(legacy,
        Dict(:step => 1234, :points => rand(Float32, 3, 10)))
    old = GaussianSplatting.load_checkpoint(legacy)
    @test old[:step] == 1234
    @test size(old[:points]) == (3, 10)
end

@testset "PLY export/import" begin
    n = 8
    gs = GaussianSplatting.GaussianModel(
        adapt(kab, rand(Float32, 3, n)),
        adapt(kab, rand(Float32, 3, 1, n)),
        # Distinct values, so a transposed `f_rest` cannot pass unnoticed.
        adapt(kab, reshape(Float32.(1:(3 * 15 * n)), 3, 15, n)),
        adapt(kab, rand(Float32, 3, n)),
        adapt(kab, rand(Float32, 4, n)),
        adapt(kab, rand(Float32, 1, n)),
        nothing, 3, 3)

    path = joinpath(mktempdir(), "splat.ply")
    GaussianSplatting.export_ply(gs, path)

    # The header external viewers parse: canonical `float` (not PlyIO's
    # `float32`) & the reference implementation's property set.
    header = String[]
    open(path) do io
        while (line = readline(io)) != "end_header"
            push!(header, line)
        end
    end
    @test !any(l -> occursin("float32", l), header)
    @test header[1] == "ply"
    @test "element vertex $n" in header
    for name in ("x", "nx", "f_dc_0", "f_rest_0", "f_rest_44", "opacity",
        "scale_0", "rot_3")
        @test "property float $name" in header
    end
    @test count(l -> startswith(l, "property"), header) == 62

    ply = GaussianSplatting.PlyIO.load_ply(path)
    vertex = ply["vertex"]
    features_rest = Array(gs.features_rest)
    # `f_rest` is channel-major in the file: R's coefficients, then G's, then B's.
    @test vertex["f_rest_0"][1] == features_rest[1, 1, 1]
    @test vertex["f_rest_14"][1] == features_rest[1, 15, 1]
    @test vertex["f_rest_15"][1] == features_rest[2, 1, 1]
    @test vertex["f_rest_30"][1] == features_rest[3, 1, 1]

    (; gaussians) = GaussianSplatting.import_ply(path, kab)
    @test Array(gaussians.points) == Array(gs.points)
    @test Array(gaussians.features_dc) == Array(gs.features_dc)
    @test Array(gaussians.features_rest) == features_rest
    @test Array(gaussians.scales) == Array(gs.scales)
    @test Array(gaussians.rotations) == Array(gs.rotations)
    @test Array(gaussians.opacities) == Array(gs.opacities)
    @test gaussians.max_sh_degree == 3

    # Degree 0: no `f_rest_*` properties at all.
    gs0 = GaussianSplatting.GaussianModel(
        adapt(kab, rand(Float32, 3, n)),
        adapt(kab, rand(Float32, 3, 1, n)),
        adapt(kab, Array{Float32}(undef, 3, 0, n)),
        adapt(kab, rand(Float32, 3, n)),
        adapt(kab, rand(Float32, 4, n)),
        adapt(kab, rand(Float32, 1, n)),
        nothing, 0, 0)
    path0 = joinpath(mktempdir(), "splat0.ply")
    GaussianSplatting.export_ply(gs0, path0)

    (; gaussians) = GaussianSplatting.import_ply(path0, kab)
    @test gaussians.max_sh_degree == 0
    @test size(gaussians.features_rest) == (3, 0, n)
    @test Array(gaussians.features_dc) == Array(gs0.features_dc)
end

end
