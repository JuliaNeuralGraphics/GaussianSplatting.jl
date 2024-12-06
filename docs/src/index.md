# GaussianSplatting.jl

Gaussian Splatting algorithm in pure Julia.

![](res/bicycle.gif)

## Requirements

- Julia 1.10 or higher.
- AMD ([AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)) or
  Nvidia ([CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)) capable machine.

## Install

Add GaussianSplatting.jl package:

```julia
] add https://github.com/JuliaNeuralGraphics/GaussianSplatting.jl.git
```

## Usage

GaussianSplatting.jl comes with a GUI application to train & view the gaussians.

1. Add necessary packages:
   ```julia
   ] add AMDGPU      # for AMD GPU
   ] add CUDA, cuDNN # for Nvidia GPU
   ] add Flux
   ```

2. Run:
   ```julia
   julia> using AMDGPU      # for AMD GPU
   julia> using CUDA, cuDNN # for Nvidia GPU
   julia> using Flux, GaussianSplatting

   julia> GaussianSplatting.gui("path-to-colmap-dataset-directory"; scale=1)
   ```

## Viewer mode

Once you've trained a model and saved it to `.bson` file you can open it
in a viewer-only mode by providing its path.

```julia
julia> GaussianSplatting.gui("path-to-checkpoint.bson")
```

Alternative, you can load a checkpoint
in a training mode (see **Usage** section) using "Save/Load" tab.

## GPU selection

This is required only the first time per the environment.
After selecting GPU backend, restart Julia REPL.

- AMD GPU:
  ```julia
  julia> Flux.gpu_backend!("AMDGPU")
  ```

- Nvidia GPU:
  ```julia
  julia> Flux.gpu_backend!("CUDA")
  ```

## References

- 3D Gaussian Splatting for Real-Time Radiance Field Rendering: <https://arxiv.org/abs/2308.04079>
- gsplat: <https://github.com/nerfstudio-project/gsplat>
