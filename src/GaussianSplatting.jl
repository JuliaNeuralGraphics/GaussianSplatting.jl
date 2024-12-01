# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
module GaussianSplatting

using Adapt
using ChainRulesCore
using Dates
using Distributions
using GPUArraysCore: @allowscalar
using GPUCompiler: @device_code
using KernelAbstractions
using KernelAbstractions: @atomic
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using NearestNeighbors
using Quaternions
using Random
using Rotations
using StaticArrays
using Statistics
using Preferences
using ImageCore
using ImageIO
using ImageTransformations
using FileIO
using VideoIO
using Zygote

using NeuralGraphicsGL
using ModernGL
using CImGui
using GLFW

import CImGui.lib as iglib

import BSON
import NNlib
import Flux
import ImageFiltering
import KernelAbstractions as KA
import NerfUtils as NU
import NeuralGraphicsGL as NGL
import SIMD
import PlyIO

const Maybe{T} = Union{T, Nothing}

struct Literal{T} end
Base.:(*)(x, ::Type{Literal{T}}) where {T} = T(x)
const u32 = Literal{UInt32}
const i32 = Literal{Int32}

const BLOCK::SVector{2, Int32} = SVector{2, Int32}(16i32, 16i32)
const BLOCK_SIZE::Int32 = 256i32

_as_T(T, x) = reinterpret(T, reshape(x, :))

include("simd.jl")
include("utils.jl")
include("metrics.jl")
include("camera.jl")
include("camera_opt.jl")
include("dataset.jl")

include("gaussians.jl")
include("rasterization/rasterizer.jl")
include("training.jl")
include("gui/gui.jl")

# Hacky way to get KA.Backend.
gpu_backend() = get_backend(Flux.gpu(Array{Int}(undef, 0)))

record_memory(kab) = return false

record_memory!(kab, v::Bool; kwargs...) = return

remove_record!(kab, x) = return

allocate_pinned(kab, T, shape) = error("Pinned memory not supported for `$kab`.")

unpin_memory(x) = error("Unpinning memory is not supported for `$(typeof(x))`.")

use_ak(kab) = false

function main(dataset_path::String; scale::Int, save_path::Maybe{String} = nothing)
    kab = gpu_backend()
    @info "Using `$kab` GPU backend."

    dataset = ColmapDataset(kab, dataset_path;
        scale, train_test_split=0.9, permute=false)
    camera = dataset.test_cameras[1]

    opt_params = OptimizationParams()
    gaussians = GaussianModel(dataset.points, dataset.colors, dataset.scales; max_sh_degree=3)
    rasterizer = GaussianRasterizer(kab, camera;
        antialias=false, fused=true, mode=:rgb)
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params)

    @info "Dataset resolution: $(Int.(camera.intrinsics.resolution))"
    @info "N train images: $(length(dataset.train_cameras))"
    @info "N test images: $(length(dataset.test_cameras))"

    t1 = time()
    for i in 1:7000
        loss = step!(trainer)
        if i == 3000
            trainer.densify = false
        end

        if trainer.step % 100 == 0 || trainer.step == 1
            image_features = rasterizer(
                gaussians.points, gaussians.opacities, gaussians.scales,
                gaussians.rotations, gaussians.features_dc, gaussians.features_rest;
                camera, sh_degree=gaussians.sh_degree)

            host_image_features = Array(image_features)
            save("image-$(trainer.step).png",
                to_image(@view(host_image_features[1:3, :, :])))

            if rasterizer.mode == :rgbd
                depth_image = permutedims(host_image_features[4, :, :], (2, 1))
                depth_image ./= 50f0 # maximum(depth_image)
                clamp01!(depth_image)
                save("depth-$(trainer.step).png", colorview(Gray, depth_image))
            end

            (; eval_ssim, eval_mse, eval_psnr) = validate(trainer)
            loss, eval_ssim, eval_mse, eval_psnr = round.(
                (loss, eval_ssim, eval_mse, eval_psnr); digits=4)
            println("i=$i | ↓ loss=$loss | ↑ ssim=$eval_ssim | ↓ mse=$eval_mse | ↑ psnr=$eval_psnr")
        end
    end
    t2 = time()
    println("Time took: $((t2 - t1) / 60) minutes.")

    if save_path ≢ nothing
        save_state(trainer, save_path)
        @info "Saved at: `$save_path`."
    end
    return
end

function gui(
    dataset_path::String; scale::Int, fullscreen::Bool = false,
)
    width, height, resizable = fullscreen ?
        (-1, -1, false) :
        (1024, 1024, true)

    gui = GSGUI(dataset_path, scale; width, height, fullscreen, resizable)
    gui |> launch!
    return
end

function gui(model_path::String, camera::Camera; fullscreen::Bool = false)
    width, height, resizable = fullscreen ?
        (-1, -1, false) :
        (1024, 1024, true)

    gaussians = GaussianModel(gpu_backend())
    θ = BSON.load(model_path)
    set_from_bson!(gaussians, θ[:gaussians])

    gui = GSGUI(gaussians, camera; width, height, fullscreen, resizable)
    gui |> launch!
    return
end

end
