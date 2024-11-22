# GaussianSplatting.jl

3D Gaussian Splatting for Real-Time Radiance Field Rendering in Julia

https://github.com/JuliaNeuralGraphics/GaussianSplatting.jl/assets/17990405/0b77d4d8-3ecb-450b-8d0d-fab2834411a7

## Requirements

- Julia 1.10.
- AMD GPU (ROCm) or Nvidia (CUDA) capable machine.

## Features

||`fused=true`|`fused=false`|
|-|-|-|
|RGB|+|+|
|Depth rendering (`mode=:rgbd`)|+|+|
|Differentiable depth|+|+|
|Differentiable camera pose|`in-progress`|`in-progress`|
|Antialiasing|`in-progress`|+|

## FYI

- Because Julia is JIT-compiled, first iteration will take longer because of kernels compilation.
  In GUI you may even see that an application is unresponsive, so you should just wait a bit.
- Julia relies on GC that is not GPU-aware, excessive allocations will impact the performance.
  To free GPU arrays we have to perform GC-sweep (in contrast to refcounting in Python) which is significantly slower.
  Because of that, you'll see significantly higher memory usage (compared to Python), but that is because of delayed freeing, not because the algorithm requires more memory.

## Usage

0. Install GaussianSplatting.jl package:

```julia
] add https://github.com/JuliaNeuralGraphics/GaussianSplatting.jl.git
```

- AMD GPU:

1. Add necessary packages: `] add AMDGPU`

2. Run:
```julia
julia> using AMDGPU, GaussianSplatting

julia> GaussianSplatting.gui("path-to-colmap-dataset-directory"; scale=1)
```

- Nvidia GPU:

1. Add necessary packages: `] add CUDA, cuDNN`

2. Run:
```julia
julia> using CUDA, cuDNN, GaussianSplatting

julia> GaussianSplatting.gui("path-to-colmap-dataset-directory"; scale=1)
```

## GPU selection

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

## Datasets

Download one of the reference datasets from the MIP-NeRF-360:
https://jonbarron.info/mipnerf360/
