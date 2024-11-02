# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
module GaussianSplatting

using VideoIO

using Adapt
using ChainRulesCore
using Dates
using Distributions
using GPUArraysCore: @allowscalar
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
using Zygote

using NeuralGraphicsGL
using ModernGL
using CImGui
using GLFW

import CImGui.lib as iglib

import BSON
import Flux
import ImageFiltering
import KernelAbstractions as KA
import NerfUtils as NU
import NeuralGraphicsGL as NGL
import PlyIO

const Maybe{T} = Union{T, Nothing}

struct Literal{T} end
Base.:(*)(x, ::Type{Literal{T}}) where {T} = T(x)
const u32 = Literal{UInt32}
const i32 = Literal{Int32}

const BLOCK::SVector{2, Int32} = SVector{2, Int32}(16i32, 16i32)
const BLOCK_SIZE::Int32 = 256i32

_as_T(T, x) = reinterpret(T, reshape(x, :))

include("utils.jl")
include("camera.jl")
include("dataset.jl")

include("gaussians.jl")
include("rasterization/rasterizer.jl")
include("training.jl")
include("gui/gui.jl")

# Hacky way to get KA.Backend.
gpu_backend() = get_backend(Flux.gpu(Array{Int}(undef, 0)))

function main(dataset_path::String; scale::Int)
    kab = gpu_backend()
    @info "Using `$kab` GPU backend."

    dataset = ColmapDataset(kab, dataset_path; scale)
    opt_params = OptimizationParams()
    gaussians = GaussianModel(dataset.points, dataset.colors, dataset.scales; max_sh_degree=3)
    rasterizer = GaussianRasterizer(kab, dataset.cameras[1];
        antialias=true, fused=false)
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params)

    camera = dataset.cameras[1]
    @info "Dataset resolution: $(Int.(camera.intrinsics.resolution))"

    for i in 1:3000
        loss = step!(trainer)
        @show i, loss

        if trainer.step % 100 == 0 || trainer.step == 1
            shs = isempty(gaussians.features_rest) ?
                gaussians.features_dc :
                hcat(gaussians.features_dc, gaussians.features_rest)
            image = rasterizer(
                gaussians.points, gaussians.opacities, gaussians.scales,
                gaussians.rotations, shs; camera, sh_degree=gaussians.sh_degree)
            save("image-$(trainer.step).png", to_image(image))

            GC.gc(false)
            GC.gc(true)
        end
    end

    # save_state(trainer, "../state.bson")
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
