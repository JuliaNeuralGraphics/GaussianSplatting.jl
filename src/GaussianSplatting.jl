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
using TOML
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
using NativeFileDialog

import CImGui.lib as iglib

import AcceleratedKernels as AK
import ChainRulesCore as CRC
import ImageFiltering
import ImPlot
import KernelAbstractions as KA
import NerfUtils as NU
import NeuralGraphicsGL as NGL
import SIMD
import PlyIO
import ProgressMeter
import SafeTensors

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
include("checkpoint.jl")
include("fused_ssim.jl")
include("camera.jl")
include("camera_opt.jl")
include("dataset.jl")
include("depth_supervision.jl")

include("gaussians.jl")
include("bilateral_grid.jl")
include("geometry_regularization.jl")
include("strategy.jl")
include("densification.jl")
include("mcmc.jl")
include("rasterization/rasterizer.jl")
include("sky_dome.jl")
include("params_io.jl")
include("training.jl")
include("gui/gui.jl")

base_array_type(backend) = error("Not implemented for backend: `$backend`.")

allocate_pinned(kab, T, shape) = error("Pinned memory not supported for `$kab`.")

"""
Synchronize `kab` without going through Julia's event loop.

GPU backends synchronize non-blockingly by default: they wait on a libuv
callback, which only the **main** thread services.
That is fine on the main thread, but when the render worker (`RenderWorker`) waits that way
it stalls until the GUI's next frame - one full vsync interval (~17ms) per synchronization,
dwarfing the render itself.
Backends that can wait without the event loop override this.

Reserve it for waits long enough to escape the backend's spin-wait budget, i.e.
the frame readback: measured on the worker, 3.6ms/frame blocking vs 19.1ms
nonblocking. Training steps synchronize within the spin and see no benefit
(6.7 vs 7.4ms/step), so they stay on the default path - a thread blocked in the
driver never reaches a GC safepoint and would stall the UI thread instead.
"""
blocking_synchronize(kab) = KA.synchronize(kab)

unpin_memory(x) = error("Unpinning memory is not supported for `$(typeof(x))`.")

use_ak(kab) = false

# If `true`, then check for NaN values in loss / gradient / params during training.
# Set via GSP_DEBUG=1 env variable.
const GSP_DEBUG::Ref{Bool} = Ref(false)

function maybe_debug()
    haskey(ENV, "GSP_DEBUG") || return

    gsp_debug = parse(Bool, ENV["GSP_DEBUG"])
    GSP_DEBUG[] = gsp_debug
    gsp_debug && @info "`GSP_DEBUG` is set to `true`. Performing extra checks for NaN values."
    return
end

function main(kab, dataset_path::String;
    scale::Int, save_path::Maybe{String} = nothing,
    opt_params::OptimizationParams = OptimizationParams(),
    strategy::Symbol = :default,
    seed::Maybe{Int} = nothing,
)
    maybe_debug()
    seed ≡ nothing || Random.seed!(seed)
    @info "Using `$kab` GPU backend."

    dataset = ColmapDataset(dataset_path; scale)
    camera = dataset.test_cameras[1]

    gaussians = GaussianModel(kab,
        dataset.points, dataset.colors, dataset.scales;
        max_sh_degree=3, isotropic=false)
    rasterizer = GaussianRasterizer(kab, camera;
        mode=training_rasterizer_mode(opt_params))

    trainer = Trainer(rasterizer, gaussians, dataset, opt_params;
        strategy=create_strategy(strategy, gaussians))

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
    for i in 1:10000
        loss = step!(trainer)
        if i == 3000
            trainer.densify = false
        end

        if trainer.step % 100 == 0 || trainer.step == 1
            image_features = rasterizer(
                gaussians.points, gaussians.opacities, gaussians.scales,
                gaussians.rotations, gaussians.features_dc, gaussians.features_rest;
                camera, sh_degree=gaussians.sh_degree, uncertainties)

            # uncertainties_h = Array(uncertainties)
            # uncertainties_image = colorview(Gray, clamp01!(transpose(uncertainties_h)))
            # save("uncertainties-$(trainer.step).png", uncertainties_image)
            # fill!(uncertainties, 0f0)

            # host_image_features = Array(image_features)
            # save("image-$(trainer.step).png",
            #     to_image(@view(host_image_features[1:3, :, :])))
            # write(writer, RGB{N0f8}.(to_image(@view(host_image_features[1:3, :, :]))))

            # if rasterizer.mode == :rgbd
            #     depth_image = to_depth(host_image_features[4, :, :])
            #     save("depth-$(trainer.step).png", depth_image)
            # end

            (; eval_ssim, eval_mse, eval_psnr) = validate(trainer)
            loss, eval_ssim, eval_mse, eval_psnr = round.(
                (loss, eval_ssim, eval_mse, eval_psnr); digits=4)
            println("i=$i | N Gaussians: $(length(gaussians)) | ↓ loss=$loss | ↑ ssim=$eval_ssim | ↓ mse=$eval_mse | ↑ psnr=$eval_psnr")
            # Weighted contributions, so they sum to `loss` & a mis-scaled
            # weight shows up as a term that dwarfs the ones it competes with.
            # The average is the one to read for trends: each step scores a
            # different view, so single-step values swing on view difficulty.
            println("        terms: $(format_breakdown(trainer.losses.current))")
            println("        ema:   $(format_breakdown(smoothed(trainer.losses)))")
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

"""
Optimization params of the reference 3DGS implementation: photometric loss
only & a fixed background. Everything this package adds on top is off, so a
run with these is comparable to published numbers.
"""
reference_opt_params(; kwargs...) = OptimizationParams(;
    use_depth_loss=false, use_bilateral_grid=false, use_normal_loss=false,
    random_background=false, kwargs...)

# Scenes of the Mip-NeRF 360 dataset & the resolution the 3DGS paper evaluates
# them at: outdoor scenes at 1/4, indoor at 1/2.
const MIPNERF360_SCALES = Dict(
    "bicycle" => 4, "flowers" => 4, "garden" => 4, "stump" => 4, "treehill" => 4,
    "bonsai" => 2, "counter" => 2, "kitchen" => 2, "room" => 2)

"""
Downscale factor the 3DGS paper evaluates the scene at `dataset_path` at.

Mip-NeRF 360 scenes are recognized by directory name; everything else
(Tanks&Temples, Deep Blending) is evaluated at the resolution it ships in.
"""
standard_scale(dataset_path::String) =
    get(MIPNERF360_SCALES, basename(normpath(dataset_path)), 1)

"""
Train & evaluate under the protocol the 3DGS paper &
[nerfbaselines](https://github.com/nerfbaselines/nerfbaselines) use, so the
numbers are comparable to published ones:

- Test split: views ordered by image filename, every 8-th held out
  (`llffhold=8`), the rest used for training. Deterministic — no random split.
- 30 000 steps, with metrics also reported at 7 000, the two checkpoints the
  paper tabulates. Densification is left to the strategy (steps 500..15 000
  for `:default`), not cut short.
- Resolution: 1/4 for Mip-NeRF 360 outdoor scenes, 1/2 for indoor
  (see [`standard_scale`](@ref)); pass `scale` to override.
- Metrics: SSIM/PSNR per test view, averaged over views, on renders rounded to
  8-bit sRGB & composited over a black background.

Multiple `configs` can be compared on the same protocol; each trains from
scratch with the same `seed`, so all of them see the same initial Gaussians &
the same train view order (until the RNG streams diverge through
strategy-specific sampling). The first config is the reference one.

Depth-supervised configs silently fall back to plain training when the dataset
has no depth priors (the `depth` column in the report tells which was actually
used).

Known deviations from the reference implementation, both of which move the
numbers a little:

- LPIPS is not reported (needs a pretrained VGG/AlexNet).
- Images are resampled up to a multiple of 16, a rasterizer requirement, so
  both render & ground truth are up to ~1% larger than the reference's.
"""
function benchmark(kab, dataset_path::String;
    scale::Maybe{Int} = nothing,
    n_steps::Int = 30_000,
    eval_at = (7_000, 30_000),
    holdout::Int = 8,
    seed::Int = 42,
    progress::Bool = true,
    # TODO add geometry_regularization
    configs = [
        (name="3dgs",                    strategy=:default, opt_params=reference_opt_params()),
        # (name="default+bilateral",       strategy=:default, opt_params=OptimizationParams(; use_depth_loss=false, use_bilateral_grid=true)),
        # (name="default+depth",           strategy=:default, opt_params=OptimizationParams(; use_depth_loss=true, use_bilateral_grid=false)),
        # (name="default+depth+bilateral", strategy=:default, opt_params=OptimizationParams(; use_depth_loss=true, use_bilateral_grid=true)),
        # (name="default+normal",          strategy=:default, opt_params=OptimizationParams(; use_depth_loss=false, use_normal_loss=true)),
        # (name="mcmc",                    strategy=:mcmc,    opt_params=OptimizationParams(; use_depth_loss=false)),
        # (name="mcmc+depth",              strategy=:mcmc,    opt_params=OptimizationParams(; use_depth_loss=true)),
    ],
)
    maybe_debug()
    @info "Using `$kab` GPU backend."
    Random.seed!(seed)

    scale ≡ nothing && (scale = standard_scale(dataset_path))
    # The dataset is read-only during training: load once, share across runs.
    # `max_extent=Inf32`: the reference implementation does not clamp it &
    # the extent scales the position LR and the densification thresholds.
    dataset = ColmapDataset(dataset_path; scale, holdout, max_extent=Inf32)
    isempty(dataset.test_cameras) && throw(ArgumentError(
        "Evaluation needs a test split, but `holdout=$holdout` left none."))
    camera = dataset.test_cameras[1]
    @info "$(length(dataset.train_cameras)) train / $(length(dataset.test_cameras)) test views " *
        "at $(Int.(camera.intrinsics.resolution)) (scale=$scale)."

    # Report at every requested step that the run actually reaches, plus the
    # last one, so a shortened run is still tabulated.
    eval_steps = sort!(unique!(Int[s for s in eval_at if s ≤ n_steps]))
    push!(eval_steps, n_steps)
    unique!(eval_steps)

    results = NamedTuple[]
    for config in configs
        @info "Benchmarking `$(config.name)`..."
        Random.seed!(seed)

        gaussians = GaussianModel(kab,
            dataset.points, dataset.colors, dataset.scales;
            max_sh_degree=3, isotropic=false)
        rasterizer = GaussianRasterizer(kab, camera;
            mode=training_rasterizer_mode(config.opt_params))
        trainer = Trainer(rasterizer, gaussians, dataset, config.opt_params;
            strategy=create_strategy(config.strategy, gaussians))
        use_depth = !isempty(trainer.depth_anchors)
        # Surface async faults from construction kernels here,
        # not at some later unrelated call.
        KA.synchronize(kab)

        # Metrics of the last checkpoint, carried between steps so the meter
        # keeps showing them until the next evaluation refreshes them.
        loss, eval_ssim, eval_psnr = 0f0, 0f0, 0f0
        train_time = 0.0
        meter = ProgressMeter.Progress(n_steps;
            desc="[$(config.name)] ", showspeed=true, enabled=progress)
        for i in 1:n_steps
            # `step!` syncs on the loss transfer, so `@elapsed` covers the
            # whole step; evaluation & progress reporting below are excluded
            # from the timing.
            train_time += @elapsed loss = step!(trainer)

            if i in eval_steps
                (; eval_ssim, eval_mse, eval_psnr) = validate(trainer; quantize=true)
                push!(results, (;
                    name=config.name, step=i, depth=use_depth,
                    minutes=train_time / 60, n_gaussians=length(gaussians), loss,
                    ssim=eval_ssim, mse=eval_mse, psnr=eval_psnr))
            end

            # A thunk: only evaluated on the steps the meter actually redraws.
            ProgressMeter.next!(meter; showvalues=() -> [
                ("gaussians", length(gaussians)),
                ("↓ loss", round(loss; digits=4)),
                ("↑ ssim", round(eval_ssim; digits=4)),
                ("↑ psnr", round(eval_psnr; digits=2))])
        end
        ProgressMeter.finish!(meter)

        # TODO (AMDGPU.jl) bulk-freeing in finalizers triggers use-after-free?
        # Release GPU memory before the next run. Synchronize around the
        # frees: no in-flight kernel may touch the buffers being released &
        # any async fault from this run surfaces here, inside its config.
        KA.synchronize(kab)
        GPUArrays.unsafe_free!(trainer.cache)
        trainer, rasterizer, gaussians = nothing, nothing, nothing
        GC.gc()
        KA.synchronize(kab)
    end

    println("\nDataset: `$dataset_path` (scale=$scale), holdout=$holdout, seed=$seed.")
    print_results(results)
    return results
end

function print_results(results)
    println(
        rpad("config", 16), rpad("step", 8), rpad("depth", 7), rpad("minutes", 9),
        rpad("gaussians", 11), rpad("↓ loss", 9), rpad("↑ ssim", 9),
        rpad("↓ mse", 10), "↑ psnr")
    for r in results
        println(
            rpad(r.name, 16), rpad(r.step, 8), rpad(r.depth, 7),
            rpad(round(r.minutes; digits=2), 9), rpad(r.n_gaussians, 11),
            rpad(round(r.loss; digits=4), 9), rpad(round(r.ssim; digits=4), 9),
            rpad(round(r.mse; digits=6), 10), round(r.psnr; digits=2))
    end
    return
end

"""
Run [`benchmark`](@ref) over every scene under `root` & report the per-scene
table plus the average over scenes - the form the 3DGS paper reports.

`scenes` are directory names under `root`; each is evaluated at its
[`standard_scale`](@ref). By default only the reference config is trained, so
the averages line up with the paper's table.
"""
function benchmark_scenes(kab, root::String;
    scenes = sort!(collect(keys(MIPNERF360_SCALES))),
    configs = [(name="3dgs", strategy=:default, opt_params=reference_opt_params())],
    kwargs...,
)
    results = NamedTuple[]
    # No scene-level bar: `benchmark` already draws a per-step one & two
    # meters would fight over the same terminal line.
    for (i, scene) in enumerate(scenes)
        scene_path = joinpath(root, scene)
        isdir(scene_path) || (@warn "Skipping missing scene `$scene_path`."; continue)

        @info "Scene $i/$(length(scenes)): `$scene`."
        for r in benchmark(kab, scene_path; configs, kwargs...)
            push!(results, (; scene, r...))
        end
    end

    println("\nRoot: `$root`.")
    for config in configs, step in unique(r.step for r in results)
        rows = filter(r -> r.name == config.name && r.step == step, results)
        isempty(rows) && continue

        println("\n`$(config.name)` @ $step steps:")
        println(
            rpad("scene", 12), rpad("minutes", 9), rpad("gaussians", 11),
            rpad("↑ ssim", 9), "↑ psnr")
        for r in rows
            println(
                rpad(r.scene, 12), rpad(round(r.minutes; digits=2), 9),
                rpad(r.n_gaussians, 11), rpad(round(r.ssim; digits=4), 9),
                round(r.psnr; digits=2))
        end
        println(
            rpad("average", 12), rpad(round(mean(r.minutes for r in rows); digits=2), 9),
            rpad(round(Int, mean(r.n_gaussians for r in rows)), 11),
            rpad(round(mean(r.ssim for r in rows); digits=4), 9),
            round(mean(r.psnr for r in rows); digits=2))
    end
    return results
end

"""
Application entry point: starts with an empty scene,
datasets are loaded via the `File` menu.

Training & rasterization run on a background thread (see
`RenderWorker`), so Julia must be started with at least 2 threads,
e.g. `julia -t 2,1` or `julia -t auto`.
"""
function app(kab; fullscreen::Bool = false)
    maybe_debug()
    width, height, resizable = fullscreen ?
        (-1, -1, false) :
        (1024, 1024, true)

    fov = NU.fov2focal(1024, 45f0)
    camera = Camera(; fx=fov, fy=fov, width, height)

    # No gaussians on startup: instantiated when loading a dataset/model.
    gui = GSGUI(kab, nothing, camera; width, height, fullscreen, resizable)
    gui |> launch!
    return
end

end
