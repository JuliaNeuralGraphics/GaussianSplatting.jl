import GaussianSplatting as GSP
import ProgressMeter

function benchmark(kab, dataset_path::String; scale::Int)
    @info "Using `$kab` GPU backend."

    dataset = GSP.ColmapDataset(dataset_path; scale, holdout=0)
    camera = dataset.train_cameras[1]
    @info "Dataset resolution: $(Int.(camera.intrinsics.resolution))"

    opt_params = GSP.OptimizationParams()
    gaussians = GSP.GaussianModel(kab,
        dataset.points, dataset.colors, dataset.scales)
    rasterizer = GSP.GaussianRasterizer(kab, camera;
        mode=:rgb)
    trainer = GSP.Trainer(rasterizer, gaussians, dataset, opt_params)

    println("Benchmarking `$dataset_path` dataset at `$scale` scale.")
    warmup_steps = 500
    n_steps = 1000

    println("Warmup for `$warmup_steps` steps:")
    warmup_meter = ProgressMeter.Progress(warmup_steps; desc="warmup ", showspeed=true)
    @time for i in 1:warmup_steps
        GSP.step!(trainer)
        ProgressMeter.next!(warmup_meter;
            showvalues=() -> [("gaussians", length(gaussians))])
    end

    println("Benchmark for `$n_steps` steps:")
    meter = ProgressMeter.Progress(n_steps; desc="benchmark ", showspeed=true)
    @time for i in 1:n_steps
        GSP.step!(trainer)
        ProgressMeter.next!(meter;
            showvalues=() -> [("gaussians", length(gaussians))])
    end
    return
end
benchmark(ROCBackend(), "/home/pxlth/Downloads/360_v2/bicycle"; scale=4)
# benchmark(CUDABackend(), "/home/pxl-th/Downloads/360_v2/bicycle"; scale=4)
