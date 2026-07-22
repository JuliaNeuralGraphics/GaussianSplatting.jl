# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

# ENV["GSP_TEST_AMDGPU"] = true
# ENV["GSP_TEST_CUDA"] = true

# import Pkg
# if get(ENV, "GSP_TEST_AMDGPU", "false") == "true"
#     @info "`GSP_TEST_AMDGPU` is `true`, importing AMDGPU.jl."
#     Pkg.add("AMDGPU")
#     using AMDGPU
#
#     kab = ROCBackend()
# elseif get(ENV, "GSP_TEST_CUDA", "false") == "true"
#     @info "`GSP_TEST_CUDA` is `true`, importing CUDA.jl."
#     Pkg.add(["CUDA", "cuDNN"])
#     using CUDA, cuDNN
#
#     kab = CUDABackend()
# else
#     error("No GPU backend was specified.")
# end

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

kab = KA.CPU()

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

# @testset "Tile ranges" begin
#     gaussian_keys = adapt(kab,
#         UInt64[0 << 32, 0 << 32, 1 << 32, 2 << 32, 3 << 32])
#
#     ranges = KA.allocate(kab, UInt32, 2, 4)
#     fill!(ranges, 0u32)
#
#     GaussianSplatting.identify_tile_range!(kab, 256)(
#         ranges, gaussian_keys; ndrange=length(gaussian_keys))
#     @test Array(ranges) == UInt32[0; 2;; 2; 3;; 3; 4;; 4; 5;;]
# end
#
# @testset "SSIM" begin
#     ssim = SSIM(kab)
#
#     x = KA.ones(kab, Float32, (16, 16, 3, 1))
#     ref = KA.zeros(kab, Float32, (16, 16, 3, 1))
#     @test ssim(x, ref) ≈ 0f0 atol=1f-4 rtol=1f-4
#     ref = KA.ones(kab, Float32, (16, 16, 3, 1))
#     @test ssim(x, ref) ≈ 1f0
#
#     x = zeros(Float32, (16, 16, 3, 1))
#     x[1:4, 1:4, :, :] .= 0.25f0
#     x[5:8, 1:4, :, :] .= 0.5f0
#     x[9:12, 13:16, :, :] .= 0.75f0
#     x[13:16, 13:16, :, :] .= 1f0
#     @test ssim(adapt(kab, x), ref) ≈ 0.1035 atol=1f-3 rtol=1f-3
#
#     x = adapt(kab, rand(Float32, 128, 128, 3, 2))
#     ref = adapt(kab, rand(Float32, 128, 128, 3, 2))
#     @test ssim(x, ref) ≈ mean(GaussianSplatting.fused_ssim(x; ref))
#
#     y, ∇ = Zygote.withgradient(x -> ssim(x, ref), x)
#     yf, ∇f = Zygote.withgradient(x -> mean(GaussianSplatting.fused_ssim(x; ref)), x)
#     @test y ≈ yf
#     @test ∇[1] ≈ ∇f[1]
# end

# @testset "Dataset loading" begin
#     dataset_dir = joinpath(@__DIR__, "..", "assets", "bicycle-smol")
#     @assert isdir(dataset_dir)

#     dataset = GaussianSplatting.ColmapDataset(kab, dataset_dir;
#         scale=8, train_test_split=1.0, permute=false, verbose=false)
#     @test length(dataset) == 6

#     @test length(dataset.train_cameras) == 6
#     cam = dataset.train_cameras[1]
#     (; width, height) = GaussianSplatting.resolution(cam)

#     img = GaussianSplatting.get_image(dataset, kab, 1, :train)
#     @test size(img, 1) == 3
#     @test size(img)[2:3] == (width, height)
#     @test KA.get_backend(img) == kab

#     global DATASET
#     if DATASET ≡ nothing
#         DATASET = dataset
#     end
# end

# @testset "Gaussians creation" begin
#     dataset_dir = joinpath(@__DIR__, "..", "assets", "bicycle-smol")
#     @assert isdir(dataset_dir)

#     global DATASET
#     dataset = if DATASET ≡ nothing
#         DATASET = GaussianSplatting.ColmapDataset(kab, dataset_dir;
#             scale=8, train_test_split=1.0, permute=false, verbose=false)
#     else
#         DATASET
#     end

#     gaussians = GaussianSplatting.GaussianModel(
#         dataset.points, dataset.colors, dataset.scales;
#         max_sh_degree=3, isotropic=false)
#     @test length(gaussians) == size(dataset.points, 2)

#     global GAUSSIANS
#     if GAUSSIANS ≡ nothing
#         GAUSSIANS = gaussians
#     end
# end

# @testset "Fused rasterizer" begin
#     global DATASET
#     global GAUSSIANS
#     dataset = DATASET
#     @assert dataset ≢ nothing
#     gaussians = GAUSSIANS
#     @assert gaussians ≢ nothing

#     camera = dataset.train_cameras[1]
#     (; width, height) = GaussianSplatting.resolution(camera)

#     rasterizer = GaussianSplatting.GaussianRasterizer(kab, camera;
#         antialias=false, fused=true, mode=:rgbd)

#     image_features = rasterizer(
#         gaussians.points, gaussians.opacities, gaussians.scales,
#         gaussians.rotations, gaussians.features_dc, gaussians.features_rest;
#         camera, sh_degree=gaussians.sh_degree)
#     @test size(image_features) == (4, width, height)
# end

# @testset "Un-fused rasterizer" begin
#     global DATASET
#     global GAUSSIANS
#     dataset = DATASET
#     @assert dataset ≢ nothing
#     gaussians = GAUSSIANS
#     @assert gaussians ≢ nothing

#     camera = dataset.train_cameras[1]
#     (; width, height) = GaussianSplatting.resolution(camera)

#     rasterizer = GaussianSplatting.GaussianRasterizer(kab, camera;
#         antialias=false, fused=false, mode=:rgbd)

#     image_features = rasterizer(
#         gaussians.points, gaussians.opacities, gaussians.scales,
#         gaussians.rotations, gaussians.features_dc, gaussians.features_rest;
#         camera, sh_degree=gaussians.sh_degree)
#     @test size(image_features) == (4, width, height)
# end

# @testset "Trainer w/ fused rasterizer" begin
#     global DATASET
#     global GAUSSIANS
#     dataset = DATASET
#     @assert dataset ≢ nothing
#     gaussians = GAUSSIANS
#     @assert gaussians ≢ nothing

#     camera = dataset.train_cameras[1]
#     (; width, height) = GaussianSplatting.resolution(camera)

#     rasterizer = GaussianSplatting.GaussianRasterizer(kab, camera;
#         antialias=false, fused=true, mode=:rgbd)

#     opt_params = GaussianSplatting.OptimizationParams()
#     trainer = GaussianSplatting.Trainer(rasterizer, gaussians, dataset, opt_params)

#     loss = GaussianSplatting.step!(trainer)
#     @test loss > 0
# end

# @testset "Trainer w/ un-fused rasterizer" begin
#     global DATASET
#     global GAUSSIANS
#     dataset = DATASET
#     @assert dataset ≢ nothing
#     gaussians = GAUSSIANS
#     @assert gaussians ≢ nothing

#     camera = dataset.train_cameras[1]
#     (; width, height) = GaussianSplatting.resolution(camera)

#     rasterizer = GaussianSplatting.GaussianRasterizer(kab, camera;
#         antialias=false, fused=false, mode=:rgbd)

#     opt_params = GaussianSplatting.OptimizationParams()
#     trainer = GaussianSplatting.Trainer(rasterizer, gaussians, dataset, opt_params)

#     loss = GaussianSplatting.step!(trainer)
#     @test loss > 0
# end

end
