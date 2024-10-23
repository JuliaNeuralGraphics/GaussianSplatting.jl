# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
using Adapt
using Test
using Zygote
using LinearAlgebra
using GaussianSplatting
using StaticArrays
using Quaternions
using Rotations

using GaussianSplatting: i32, u32

import GaussianSplatting as GSP
import KernelAbstractions as KA

const Backend = GaussianSplatting.Backend

@testset "quat2mat" begin
    r = RotXYZ(rand(Float32), rand(Float32), rand(Float32))
    q = QuatRotation{Float32}(r)

    ŷ = GaussianSplatting.quat2mat(SVector{4, Float32}(q.w, q.x, q.y, q.z))
    y = SMatrix{3, 3, Float32, 9}(q)
    @test all(ŷ .≈ y)
end

@testset "get_rect" begin
    width, height = 1024, 1024
    block = SVector{2, Int32}(16, 16)
    grid = SVector{2, Int32}(cld(width, block[1]), cld(height, block[2]))

    # rect covering only one block
    rmin, rmax = GaussianSplatting.get_rect(
        SVector{2, Float32}(0, 0), 1i32, grid, block)
    @test all(rmin .== (0, 0))
    @test all(rmax .== (1, 1))

    # rect covering 2 blocks
    rmin, rmax = GaussianSplatting.get_rect(
        SVector{2, Float32}(0, 0), Int32(block[1] + 1), grid, block)
    @test all(rmin .== (0, 0))
    @test all(rmax .== (2, 2))
end

@testset "Tile ranges" begin
    kab = KA.CPU() # TODO use kab
    gaussian_keys = adapt(kab,
        UInt64[0 << 32, 0 << 32, 1 << 32, 2 << 32, 3 << 32])

    ranges = KA.allocate(kab, UInt32, 2, 4)
    fill!(ranges, 0u32)

    GaussianSplatting.identify_tile_range!(kab, 256)(
        ranges, gaussian_keys; ndrange=length(gaussian_keys))
    @test Array(ranges) == UInt32[0; 2;; 2; 3;; 3; 4;; 4; 5;;]
end

@testset "SSIM" begin
    kab = KA.CPU()
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
