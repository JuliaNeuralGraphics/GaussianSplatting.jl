# GaussianSplatting.jl

3D Gaussian Splatting for Real-Time Radiance Field Rendering in Julia

## Requirements

- Julia 1.10: https://julialang.org/downloads/
- AMD GPU capable machine with ROCm installation.
- Or Nvidia GPU capable machine.

## Usage

1. Clone/download the repo.
2. Instantiate the project from its directory:
   1. Start Julia REPL: `julia --project=. --threads=auto`.
   2. Install dependencies: `]up`.
3. Start the GUI from REPL:
```julia
julia> using GaussianSplatting
julia> GaussianSplatting.gui("path-to-colmap-dataset-directory")
```

## GPU selection

- AMD GPU:
  1. In `LocalPreferences.toml` set:
     ```toml
     [Flux]
     gpu_backend = "AMDGPU"
     [GaussianSplatting]
     backend="AMDGPU"
     ```

- Nvidia GPU:
  1. In `LocalPreferences.toml` set:
     ```toml
     [Flux]
     gpu_backend = "CUDA"
     [GaussianSplatting]
     backend="CUDA"
     ```

## Datasets

Download one of the reference datasets from the MIP-NeRF-360:
https://jonbarron.info/mipnerf360/
