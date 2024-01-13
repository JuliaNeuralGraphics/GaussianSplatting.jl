using GaussianSplatting

function main(dataset_path::String, scale::Int = 1)
    kab = GaussianSplatting.Backend
    GaussianSplatting.get_module(kab).allowscalar(false)

    dataset = GaussianSplatting.ColmapDataset(kab, dataset_path; scale)
    opt_params = GaussianSplatting.OptimizationParams()
    gaussians = GaussianSplatting.GaussianModel(
        dataset.points, dataset.colors, dataset.scales)
    rasterizer = GaussianSplatting.GaussianRasterizer(kab,
        dataset.cameras[1]; auxiliary=false)
    trainer = GaussianSplatting.Trainer(rasterizer, gaussians, dataset, opt_params)

    println("Benchmarking `$dataset_path` dataset at `$scale` scale.")
    warmup_steps = 500
    n_steps = 1000

    t1 = time()
    for i in 1:warmup_steps
        GaussianSplatting.step!(trainer)
    end
    t2 = time()
    println("Warmup `$warmup_steps` steps took $(t2 - t1) seconds.")

    t1 = time()
    for i in 1:n_steps
        GaussianSplatting.step!(trainer)
    end
    t2 = time()
    println("Benchmark `$n_steps` steps took $(t2 - t1) seconds.")
    return
end
main("/home/pxlth/Downloads/360_v2/bicycle", 4)
