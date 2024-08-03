# GaussianSplatting.jl

3D Gaussian Splatting for Real-Time Radiance Field Rendering in Julia

https://github.com/JuliaNeuralGraphics/GaussianSplatting.jl/assets/17990405/0b77d4d8-3ecb-450b-8d0d-fab2834411a7

## Requirements

- Julia 1.10.
- AMD GPU (ROCm) or Nvidia (CUDA) capable machine.

## Usage

1. Clone/download the repo.
2. Instantiate the project from its directory:
   1. Start Julia REPL: `julia --project=. --threads=auto`.
   2. Install dependencies: `]up`.
3. Start the GUI from REPL:
```julia
julia> using GaussianSplatting
julia> image_scale = 1
julia> GaussianSplatting.gui("path-to-colmap-dataset-directory", image_scale)
```

## GPU selection

- AMD GPU:
  1. In `LocalPreferences.toml` set:
     ```toml
     [Flux]
     gpu_backend = "AMDGPU"
     ```

- Nvidia GPU:
  1. In `LocalPreferences.toml` set:
     ```toml
     [Flux]
     gpu_backend = "CUDA"
     ```

## Datasets

Download one of the reference datasets from the MIP-NeRF-360:
https://jonbarron.info/mipnerf360/
