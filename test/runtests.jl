# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

# ENV["GSP_TEST_AMDGPU"] = true
# ENV["GSP_TEST_CUDA"] = true

import Pkg
if get(ENV, "GSP_TEST_AMDGPU", "false") == "true"
    @info "`GSP_TEST_AMDGPU` is `true`, importing AMDGPU.jl."
    Pkg.develop("AMDGPU")
    using AMDGPU
elseif get(ENV, "GSP_TEST_CUDA", "false") == "true"
    @info "`GSP_TEST_CUDA` is `true`, importing CUDA.jl & cuDNN.jl."
    Pkg.add("CUDA")
    Pkg.add("cuDNN")
    using CUDA, cuDNN
else
    error("No GPU backend was specified.")
end

using Adapt
using Test
using Zygote
using LinearAlgebra
using GaussianSplatting
using StaticArrays
using Quaternions
using Rotations

using GaussianSplatting: i32, u32

import KernelAbstractions as KA

DATASET = nothing
GAUSSIANS = nothing

@info "Testing on `$(GaussianSplatting.gpu_backend())` backend."

@testset "GaussianSplatting" begin

@testset "quat2mat" begin
    r = RotXYZ(rand(Float32), rand(Float32), rand(Float32))
    q = QuatRotation{Float32}(r)

    ŷ = @inferred GaussianSplatting.unnorm_quat2rot(SVector{4, Float32}(q.w, q.x, q.y, q.z))
    y = SMatrix{3, 3, Float32, 9}(q)
    @test all(ŷ .≈ y)
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

@testset "Tile ranges" begin
    kab = GaussianSplatting.gpu_backend()
    gaussian_keys = adapt(kab,
        UInt64[0 << 32, 0 << 32, 1 << 32, 2 << 32, 3 << 32])

    ranges = KA.allocate(kab, UInt32, 2, 4)
    fill!(ranges, 0u32)

    GaussianSplatting.identify_tile_range!(kab, 256)(
        ranges, gaussian_keys; ndrange=length(gaussian_keys))
    @test Array(ranges) == UInt32[0; 2;; 2; 3;; 3; 4;; 4; 5;;]
end

@testset "SSIM" begin
    kab = GaussianSplatting.gpu_backend()
    ssim = GaussianSplatting.SSIM(kab)

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
end

@testset "Dataset loading" begin
    kab = GaussianSplatting.gpu_backend()
    dataset_dir = joinpath(@__DIR__, "..", "assets", "bicycle-smol")
    @assert isdir(dataset_dir)

    dataset = GaussianSplatting.ColmapDataset(kab, dataset_dir;
        scale=8, train_test_split=1.0, permute=false, verbose=false)
    @test length(dataset) == 6

    @test length(dataset.train_cameras) == 6
    cam = dataset.train_cameras[1]
    (; width, height) = GaussianSplatting.resolution(cam)

    img = GaussianSplatting.get_image(dataset, kab, 1, :train)
    @test size(img, 1) == 3
    @test size(img)[2:3] == (width, height)
    @test KA.get_backend(img) == kab

    global DATASET
    if DATASET ≡ nothing
        DATASET = dataset
    end
end

@testset "Gaussians creation" begin
    kab = GaussianSplatting.gpu_backend()
    dataset_dir = joinpath(@__DIR__, "..", "assets", "bicycle-smol")
    @assert isdir(dataset_dir)

    global DATASET
    dataset = if DATASET ≡ nothing
        DATASET = GaussianSplatting.ColmapDataset(kab, dataset_dir;
            scale=8, train_test_split=1.0, permute=false, verbose=false)
    else
        DATASET
    end

    gaussians = GaussianSplatting.GaussianModel(
        dataset.points, dataset.colors, dataset.scales;
        max_sh_degree=3, isotropic=false)
    @test length(gaussians) == size(dataset.points, 2)

    global GAUSSIANS
    if GAUSSIANS ≡ nothing
        GAUSSIANS = gaussians
    end
end

@testset "Fused rasterizer" begin
    global DATASET
    global GAUSSIANS
    dataset = DATASET
    @assert dataset ≢ nothing
    gaussians = GAUSSIANS
    @assert gaussians ≢ nothing

    camera = dataset.train_cameras[1]
    (; width, height) = GaussianSplatting.resolution(camera)

    kab = GaussianSplatting.gpu_backend()
    rasterizer = GaussianSplatting.GaussianRasterizer(kab, camera;
        antialias=false, fused=true, mode=:rgbd)

    image_features = rasterizer(
        gaussians.points, gaussians.opacities, gaussians.scales,
        gaussians.rotations, gaussians.features_dc, gaussians.features_rest;
        camera, sh_degree=gaussians.sh_degree)
    @test size(image_features) == (4, width, height)
end

@testset "Un-fused rasterizer" begin
    global DATASET
    global GAUSSIANS
    dataset = DATASET
    @assert dataset ≢ nothing
    gaussians = GAUSSIANS
    @assert gaussians ≢ nothing

    camera = dataset.train_cameras[1]
    (; width, height) = GaussianSplatting.resolution(camera)

    kab = GaussianSplatting.gpu_backend()
    rasterizer = GaussianSplatting.GaussianRasterizer(kab, camera;
        antialias=false, fused=false, mode=:rgbd)

    image_features = rasterizer(
        gaussians.points, gaussians.opacities, gaussians.scales,
        gaussians.rotations, gaussians.features_dc, gaussians.features_rest;
        camera, sh_degree=gaussians.sh_degree)
    @test size(image_features) == (4, width, height)
end

@testset "Trainer w/ fused rasterizer" begin
    global DATASET
    global GAUSSIANS
    dataset = DATASET
    @assert dataset ≢ nothing
    gaussians = GAUSSIANS
    @assert gaussians ≢ nothing

    camera = dataset.train_cameras[1]
    (; width, height) = GaussianSplatting.resolution(camera)

    kab = GaussianSplatting.gpu_backend()
    rasterizer = GaussianSplatting.GaussianRasterizer(kab, camera;
        antialias=false, fused=true, mode=:rgbd)

    opt_params = GaussianSplatting.OptimizationParams()
    trainer = GaussianSplatting.Trainer(rasterizer, gaussians, dataset, opt_params)

    loss = GaussianSplatting.step!(trainer)
    @test loss > 0
end

# @testset "Trainer w/ un-fused rasterizer" begin
#     global DATASET
#     global GAUSSIANS
#     dataset = DATASET
#     @assert dataset ≢ nothing
#     gaussians = GAUSSIANS
#     @assert gaussians ≢ nothing

#     camera = dataset.train_cameras[1]
#     (; width, height) = GaussianSplatting.resolution(camera)

#     kab = GaussianSplatting.gpu_backend()
#     rasterizer = GaussianSplatting.GaussianRasterizer(kab, camera;
#         antialias=false, fused=false, mode=:rgbd)

#     opt_params = GaussianSplatting.OptimizationParams()
#     trainer = GaussianSplatting.Trainer(rasterizer, gaussians, dataset, opt_params)

#     loss = GaussianSplatting.step!(trainer)
#     @test loss > 0
# end

end
