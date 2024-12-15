import GaussianSplatting as GSP

function main(dataset_path::String; scale::Int)
    kab = GSP.gpu_backend()
    @info "Using `$kab` GPU backend."

    dataset = GSP.ColmapDataset(kab, dataset_path; scale,
        train_test_split=1, permute=false)
    camera = dataset.train_cameras[1]
    @info "Dataset resolution: $(Int.(camera.intrinsics.resolution))"

    opt_params = GSP.OptimizationParams()
    gaussians = GSP.GaussianModel(
        dataset.points, dataset.colors, dataset.scales)
    rasterizer = GSP.GaussianRasterizer(kab, camera;
        antialias=false, fused=true, mode=:rgb)
    trainer = GSP.Trainer(rasterizer, gaussians, dataset, opt_params)

    println("Benchmarking `$dataset_path` dataset at `$scale` scale.")
    warmup_steps = 500
    n_steps = 1000

    println("Warmup for `$warmup_steps` steps:")
    @time for i in 1:warmup_steps
        GSP.step!(trainer)
    end

    println("Benchmark for `$n_steps` steps:")
    @time for i in 1:n_steps
        GSP.step!(trainer)
    end
    return
end
main("/home/pxlth/Downloads/360_v2/bicycle"; scale=4)
