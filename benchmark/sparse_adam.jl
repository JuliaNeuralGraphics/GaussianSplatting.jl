import GaussianSplatting as GSP
import ProgressMeter

function run_steps!(trainer, gaussians, n_steps; desc)
    meter = ProgressMeter.Progress(n_steps; desc, showspeed=true)
    elapsed = @elapsed for i in 1:n_steps
        GSP.step!(trainer)
        ProgressMeter.next!(meter; showvalues=() -> [("gaussians", length(gaussians))])
    end
    return elapsed
end

function benchmark_sparse_adam(kab, dataset_path::String; scale::Int)
    @info "Using `$kab` GPU backend."

    dataset = GSP.ColmapDataset(dataset_path; scale, holdout=0)
    camera = dataset.train_cameras[1]
    @info "Dataset resolution: $(Int.(camera.intrinsics.resolution))"

    warmup_steps = 500
    n_steps = 1000

    results = Dict{Symbol, Float64}()
    for use_sparse_adam in (false, true)
        label = use_sparse_adam ? :sparse : :dense
        println("Benchmarking `$dataset_path` dataset at `$scale` scale ($label Adam).")

        opt_params = GSP.OptimizationParams(; use_sparse_adam)
        gaussians = GSP.GaussianModel(kab, dataset.points, dataset.colors, dataset.scales)
        rasterizer = GSP.GaussianRasterizer(kab, camera; mode=:rgb)
        trainer = GSP.Trainer(rasterizer, gaussians, dataset, opt_params)

        println("Warmup ($label) for `$warmup_steps` steps:")
        run_steps!(trainer, gaussians, warmup_steps; desc="warmup ($label) ")

        println("Benchmark ($label) for `$n_steps` steps:")
        results[label] = run_steps!(trainer, gaussians, n_steps; desc="benchmark ($label) ")
    end

    println("Dense Adam:  $(results[:dense]) s ($(n_steps / results[:dense]) it/s)")
    println("Sparse Adam: $(results[:sparse]) s ($(n_steps / results[:sparse]) it/s)")
    println("Speedup: $(results[:dense] / results[:sparse])x")
    return results
end

benchmark_sparse_adam(MetalBackend(), "/Users/pxlth/Downloads/360_v2/bicycle"; scale=4)
# benchmark_sparse_adam(ROCBackend(), "/home/pxlth/Downloads/360_v2/bicycle"; scale=4)
# benchmark_sparse_adam(CUDABackend(), "/home/pxl-th/Downloads/360_v2/bicycle"; scale=4)
