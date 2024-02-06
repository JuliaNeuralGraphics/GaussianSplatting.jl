module GaussianSplatting

using Adapt
using ChainRulesCore
using Distributions
using KernelAbstractions
using KernelAbstractions: @atomic
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
using CImGui.ImGuiGLFWBackend.LibGLFW
using VideoIO

import BSON
import Flux
import ImageFiltering
import KernelAbstractions as KA
import NerfUtils as NU
import NeuralGraphicsGL as NGL

const Maybe{T} = Union{T, Nothing}

struct Literal{T} end
Base.:(*)(x, ::Type{Literal{T}}) where {T} = T(x)
const u32 = Literal{UInt32}
const i32 = Literal{Int32}

const BLOCK::SVector{2, Int32} = SVector{2, Int32}(16i32, 16i32)
const BLOCK_SIZE::Int32 = 256i32

include("kautils.jl")
include("camera.jl")
include("dataset.jl")

include("gaussians.jl")
include("rasterization/rasterizer.jl")
include("training.jl")
include("gui/gui.jl")

"""
TODO
- [GUI] toggle for densification
- [GUI] reset opacity toggle

- raw visualization mode (export to PLY)
- compute depth (differentiable) & silhouette
- allow isotropic gaussians
"""

function main(dataset_path::String, scale::Int = 8)
    kab = Backend
    get_module(kab).allowscalar(false)

    cameras_file = joinpath(dataset_path, "sparse/0/cameras.bin")
    images_file = joinpath(dataset_path, "sparse/0/images.bin")
    points_file = joinpath(dataset_path, "sparse/0/points3D.bin")
    images_dir = joinpath(dataset_path, "images")

    dataset = ColmapDataset(kab;
        cameras_file, images_file, points_file, images_dir, scale)
    opt_params = OptimizationParams()
    gaussians = GaussianModel(dataset.points, dataset.colors, dataset.scales; max_sh_degree=0)
    rasterizer = GaussianRasterizer(kab, dataset.cameras[1]; auxiliary=true)
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params)

    for i in 1:3000
        loss = step!(trainer)
        @show i, loss

        if trainer.step % 100 == 0
            cam_idx = rand(1:length(trainer.dataset.cameras))
            camera = trainer.dataset.cameras[cam_idx]

            # covisibility = KA.allocate(kab, Bool, length(gaussians))
            # fill!(covisibility, false)

            shs = isempty(gaussians.features_rest) ?
                gaussians.features_dc :
                hcat(gaussians.features_dc, gaussians.features_rest)
            rasterizer(
                gaussians.points, gaussians.opacities, gaussians.scales,
                gaussians.rotations, shs; camera, sh_degree=gaussians.sh_degree,
                covisibility=nothing)

            # intr = covisibility .& covisibility
            # unio = covisibility .| covisibility
            # @show sum(intr)
            # @show sum(unio)
            # @show sum(intr) / sum(unio)

            final_img = to_image(rasterizer)
            save("image-$(trainer.step).png", final_img)

            if has_auxiliary(rasterizer)
                depth_img = to_depth(rasterizer; normalize=true)
                uncertainty_img = to_uncertainty(rasterizer)
                save("depth-$(trainer.step).png", depth_img)
                save("uncertainty-$(trainer.step).png", uncertainty_img)
            end
        end
    end
    return
end

function gui(dataset_path::String, scale::Int = 8; fullscreen::Bool = false)
    gsgui = if fullscreen
        GSGUI(dataset_path, scale; fullscreen=true, resizable=false)
    else
        GSGUI(dataset_path, scale; width=1024, height=1024, resizable=true)
    end
    gsgui |> launch!
    return
end

end
