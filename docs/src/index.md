# GaussianSplatting.jl

Gaussian Splatting algorithm in pure Julia.

![](res/bicycle.gif)

## Requirements

- Julia 1.10 or higher.
- [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) or
  [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or
  [Metal.jl](https://github.com/JuliaGPU/Metal.jl) capable machine.

## Install

Add GaussianSplatting.jl package:

```julia
] add https://github.com/JuliaNeuralGraphics/GaussianSplatting.jl.git
```

## Usage

GaussianSplatting.jl comes with a GUI application to train & view the gaussians.

1. Add necessary packages:
   ```julia
   ] add AMDGPU # for AMD GPU
   ] add CUDA   # for Nvidia GPU
   ] add Metal  # for Apple GPU
   ```

2. Start Julia with at least 2 threads (training runs on a background
   thread to keep the UI responsive):
   ```bash
   julia -t auto # or `julia -t 2,1`
   ```

3. Run:
   ```julia
   julia> using AMDGPU; kab = ROCBackend()  # for AMD GPU
   julia> using CUDA; kab = CUDABackend()   # for Nvidia GPU
   julia> using Metal; kab = MetalBackend() # for Apple GPU
   julia> GaussianSplatting.gui(kab, "path-to-colmap-dataset-directory"; scale=1)
   ```

## Hyperparameters

The `Open Dataset` dialog exposes a handful of the most useful options as
checkboxes; the rest of `OptimizationParams` (learning rates, loss weights,
warm-up iterations) is read from an optional `.toml` file next to them.

Use `Save...` to write out the values a run is about to use, and `Load...` to
train another dataset with exactly those — which is what makes a reconstruction
reproducible once the session that produced it is gone.

Fields that the file omits keep their default, so it only needs to list what a
run changes:

```toml
lr_points_start = 0.00016
use_bilateral_grid = true
bilateral_grid_size = [16, 16, 8]
depth_loss_weight = 2.0
depth_loss_mode = "ssi"
```

Unknown keys and values of the wrong type are errors, not warnings: a
hyperparameter that silently failed to apply looks exactly like one that did.
The checkboxes in the dialog override the fields they expose, and are re-synced
to the file's values when one is loaded.

The same file works outside the GUI:

```julia
julia> opt_params = GaussianSplatting.load_opt_params("params.toml")
julia> GaussianSplatting.save_opt_params("params.toml", opt_params)
```

## Viewer mode

Once you've trained a model and saved it to a `.safetensors` file you can open it
in a viewer-only mode by providing its path.

```julia
julia> GaussianSplatting.gui(kab, "path-to-checkpoint.safetensors")
```

Alternative, you can load a checkpoint
in a training mode (see **Usage** section) using "Save/Load" tab.

## References

- 3D Gaussian Splatting for Real-Time Radiance Field Rendering: <https://arxiv.org/abs/2308.04079>
- gsplat: <https://github.com/nerfstudio-project/gsplat>
