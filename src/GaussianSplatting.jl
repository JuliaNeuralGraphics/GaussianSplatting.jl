module GaussianSplatting

using Adapt
using ChainRulesCore
using Distributions
using Enzyme
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
    rasterizer = GaussianRasterizer(kab, dataset.cameras[1]; auxiliary=false)
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params)

    for i in 1:1000
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

            save("image-$(trainer.step).png", to_image(rasterizer))

            if has_auxiliary(rasterizer)
                depth_img = to_depth(rasterizer; normalize=true)
                uncertainty_img = to_uncertainty(rasterizer)
                save("depth-$(trainer.step).png", depth_img)
                save("uncertainty-$(trainer.step).png", uncertainty_img)
            end
        end
    end

    save_state(trainer, "state.bson")

    return
end

function track(dataset_path::String, scale::Int = 8)
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
    rasterizer = GaussianRasterizer(kab, dataset.cameras[1]; auxiliary=false)
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params)

    load_state!(trainer, "state.bson")

    cam_idx = rand(1:length(trainer.dataset.cameras))
    camera = trainer.dataset.cameras[cam_idx]
    target_image = get_image(trainer, cam_idx)

    # Frame for video.
    target_frame = reshape(Array(target_image), size(target_image)[1:3])
    target_frame = permutedims(target_frame, (3, 1, 2))
    target_frame = transpose(colorview(RGB, target_frame) .|> RGB{N0f8})

    w2c = camera.w2c
    qh = QuatRotation(RotXYZ(w2c[1:3, 1:3]))
    q = adapt(kab, Float32[qh.q.s, qh.q.v1, qh.q.v2, qh.q.v3])
    t = adapt(kab, w2c[1:3, 4])

    shs = isempty(gaussians.features_rest) ?
        gaussians.features_dc :
        hcat(gaussians.features_dc, gaussians.features_rest)
    rasterizer(
        gaussians.points, gaussians.opacities, gaussians.scales,
        gaussians.rotations, shs,
        q, t; camera, sh_degree=gaussians.sh_degree,
        covisibility=nothing)

    # Modify pose and start recon.
    t = adapt(kab, w2c[1:3, 4] .+ rand(Float32, 3))

    q_opt = NU.Adam(kab, q; lr=1f-5)
    t_opt = NU.Adam(kab, t; lr=1f-2)

    res = resolution(camera)
    test_frame = zeros(RGB{N0f8}, res.height, res.width * 2)
    writer = open_video_out(
        "./track.mp4", test_frame; framerate=60,
        target_pix_fmt=VideoIO.AV_PIX_FMT_YUV420P)

    for i in 1:1000
        loss, ∇ = Zygote.withgradient(q, t) do q, t
            shs = isempty(gaussians.features_rest) ?
                gaussians.features_dc :
                hcat(gaussians.features_dc, gaussians.features_rest)
            img = rasterizer(
                gaussians.points, gaussians.opacities, gaussians.scales,
                gaussians.rotations, shs,
                q, t; camera, sh_degree=gaussians.sh_degree)

            # From (c, w, h) to (w, h, c, 1) for SSIM.
            img_tmp = permutedims(img, (2, 3, 1))
            img_eval = reshape(img_tmp, size(img_tmp)..., 1)

            # Only L1 for tracking.
            mean(abs.(img_eval .- target_image))
        end
        @show i, loss

        NU.step!(q_opt, q, ∇[1]; dispose=true)
        NU.step!(t_opt, t, ∇[2]; dispose=true)

        current_frame = RGB{N0f8}.(to_image(rasterizer))
        frame = hcat(current_frame, target_frame)
        @assert size(test_frame) == size(frame)
        write(writer, frame)
    end

    close_video_out!(writer)
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
