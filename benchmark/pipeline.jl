import GaussianSplatting as GSP

function main(dataset_path::String; scale::Int)
    kab = GSP.gpu_backend()

    dataset = GSP.ColmapDataset(kab, dataset_path; scale)
    opt_params = GSP.OptimizationParams()
    gaussians = GSP.GaussianModel(
        dataset.points, dataset.colors, dataset.scales)
    rasterizer = GSP.GaussianRasterizer(kab, dataset.cameras[1])
    trainer = GSP.Trainer(rasterizer, gaussians, dataset, opt_params)

    println("Benchmarking `$dataset_path` dataset at `$scale` scale.")
    warmup_steps = 500
    n_steps = 1000

    t1 = time()
    for i in 1:warmup_steps
        GSP.step!(trainer)
    end
    t2 = time()
    println("Warmup `$warmup_steps` steps took $(t2 - t1) seconds.")

    t1 = time()
    for i in 1:n_steps
        GSP.step!(trainer)
    end
    t2 = time()
    println("Benchmark `$n_steps` steps took $(t2 - t1) seconds.")
    return
end
main("/home/pxl-th/Downloads/360_v2/bicycle"; scale=4)
