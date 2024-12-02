# GaussianSplatting.jl

Gaussian Splatting algorithm in pure Julia.

## Requirements

- Julia 1.10 or higher.
- AMD ([AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)) or
  Nvidia ([CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)) capable machine.

## Usage

1. Install GaussianSplatting.jl package:

```julia
] add https://github.com/JuliaNeuralGraphics/GaussianSplatting.jl.git
```

### For AMD GPU

2. Add necessary packages: `] add AMDGPU`

3. Run:
```julia
julia> using AMDGPU, GaussianSplatting

julia> GaussianSplatting.gui("path-to-colmap-dataset-directory"; scale=1)
```

### For Nvidia GPU

2. Add necessary packages: `] add CUDA, cuDNN`

3. Run:
```julia
julia> using CUDA, cuDNN, GaussianSplatting

julia> GaussianSplatting.gui("path-to-colmap-dataset-directory"; scale=1)
```

## GPU selection

This is required only the first time per the environment.
After selecting GPU backend, restart Julia REPL.

- AMD GPU:
  ```julia
  julia> using Flux

  julia> Flux.gpu_backend!("AMDGPU")
  ```

- Nvidia GPU:
  ```julia
  julia> using Flux

  julia> Flux.gpu_backend!("CUDA")
  ```

## References

- 3D Gaussian Splatting for Real-Time Radiance Field Rendering: <https://arxiv.org/abs/2308.04079>
- gsplat: <https://github.com/nerfstudio-project/gsplat>
