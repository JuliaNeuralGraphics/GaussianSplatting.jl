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

import Flux
import ImageFiltering
import KernelAbstractions as KA
import NerfUtils as NU
import NeuralGraphicsGL as NGL

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
- [GUI] disable resizable for capture mode
- [GUI] change rendering resolution

- compute camera extent (needed for gaussians model from PC)
- printout stats
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
    gaussians = GaussianModel(dataset.points, dataset.colors, dataset.scales)
    rasterizer = GaussianRasterizer(kab, dataset.cameras[1])
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params)

    for i in 1:1000
        loss = step!(trainer)
        @show i, loss

        if trainer.step % 1000 == 0
            camera = trainer.dataset.cameras[1]

            shs = hcat(gaussians.features_dc, gaussians.features_rest)
            rasterizer(
                gaussians.points, gaussians.opacities, gaussians.scales,
                gaussians.rotations, shs; camera, sh_degree=gaussians.sh_degree)
            final_img = to_image(rasterizer)
            save("image-$(trainer.step).png", final_img)
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
