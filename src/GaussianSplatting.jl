# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
module GaussianSplatting

using Adapt
using ChainRulesCore
using Dates
using Distributions
using GPUArraysCore: @allowscalar
using GPUArrays
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
import ChainRulesCore as CRC
import NNlib
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
include("fused_ssim.jl")
include("camera.jl")
include("camera_opt.jl")
include("dataset.jl")

include("gaussians.jl")
include("densification.jl")
include("rasterization/rasterizer.jl")
include("training.jl")
include("gui/gui.jl")

base_array_type(backend) = error("Not implemented for backend: `$backend`.")

allocate_pinned(kab, T, shape) = error("Pinned memory not supported for `$kab`.")

unpin_memory(x) = error("Unpinning memory is not supported for `$(typeof(x))`.")

use_ak(kab) = false

function main(kab, dataset_path::String; scale::Int, save_path::Maybe{String} = nothing)
    @info "Using `$kab` GPU backend."

    dataset = ColmapDataset(kab, dataset_path;
        scale, train_test_split=0.9, permute=false)
    camera = dataset.test_cameras[1]

    gaussians = GaussianModel(dataset.points, dataset.colors, dataset.scales;
        max_sh_degree=3, isotropic=false)
    rasterizer = GaussianRasterizer(kab, camera;
        antialias=false, fused=true, mode=:rgbd)

    opt_params = OptimizationParams()
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params)

    @info "Dataset resolution: $(Int.(camera.intrinsics.resolution))"
    @info "N train images: $(length(dataset.train_cameras))"
    @info "N test images: $(length(dataset.test_cameras))"

    width, height = camera.intrinsics.resolution
    uncertainties = adapt(kab, zeros(Float32, width, height))

    # res = resolution(camera)
    # writer = open_video_out(
    #     "./out.mp4", zeros(RGB{N0f8}, res.height, res.width);
    #     framerate=60, target_pix_fmt=VideoIO.AV_PIX_FMT_YUV420P)

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
                camera, sh_degree=gaussians.sh_degree, uncertainties)

            uncertainties_h = Array(uncertainties)
            uncertainties_image = colorview(Gray, clamp01!(transpose(uncertainties_h)))
            save("uncertainties-$(trainer.step).png", uncertainties_image)
            fill!(uncertainties, 0f0)

            host_image_features = Array(image_features)
            save("image-$(trainer.step).png",
                to_image(@view(host_image_features[1:3, :, :])))
            # write(writer, RGB{N0f8}.(to_image(@view(host_image_features[1:3, :, :]))))

            if rasterizer.mode == :rgbd
                depth_image = to_depth(host_image_features[4, :, :])
                save("depth-$(trainer.step).png", depth_image)
            end

            (; eval_ssim, eval_mse, eval_psnr) = validate(trainer)
            loss, eval_ssim, eval_mse, eval_psnr = round.(
                (loss, eval_ssim, eval_mse, eval_psnr); digits=4)
            println("i=$i | N Gaussians: $(length(gaussians)) | ↓ loss=$loss | ↑ ssim=$eval_ssim | ↓ mse=$eval_mse | ↑ psnr=$eval_psnr")
        end
    end
    t2 = time()
    println("Time took: $((t2 - t1) / 60) minutes.")
    # close_video_out!(writer)

    if save_path ≢ nothing
        save_state(trainer, save_path)
        @info "Saved at: `$save_path`."
    end
    return
end

function gui(kab, path::String; scale::Maybe{Int} = nothing, fullscreen::Bool = false)
    ispath(path) || error("Path does not exist: `$path`.")

    viewer_mode = endswith(path, ".bson") || endswith(path, ".ply")
    !viewer_mode && !isdir(path) && error(
        "`path` must be either a [`.bson`, '.ply'] model checkpoint or " *
        "a directory with COLMAP dataset, instead: `$path`.")
    !viewer_mode && scale ≡ nothing && error(
        "`scale` keyword argument must be specified if `path` is a COLMAP dataset.")

    width, height, resizable = fullscreen ?
        (-1, -1, false) :
        (1024, 1024, true)

    gui = if viewer_mode
        if endswith(path, ".bson")
            θ = BSON.load(path)
            gaussians = GaussianModel(kab)
            set_from_bson!(gaussians, θ[:gaussians])
            camera = θ[:camera]
        else
            (; gaussians) = import_ply(path, kab)
            width = 1024
            fov = NU.fov2focal(1024, 45f0)
            camera = Camera(; fx=fov, fy=fov, width, height=width)
        end

        GSGUI(kab, gaussians, camera; width, height, fullscreen, resizable)
    else
        GSGUI(kab, path, scale; width, height, fullscreen, resizable)
    end
    gui |> launch!
    return
end

end
