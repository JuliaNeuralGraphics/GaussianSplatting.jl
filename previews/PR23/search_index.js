var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"#GaussianSplatting.jl","page":"Home","title":"GaussianSplatting.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Gaussian Splatting algorithm in pure Julia.","category":"page"},{"location":"#Requirements","page":"Home","title":"Requirements","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Julia 1.10 or higher.\nAMD (AMDGPU.jl) or Nvidia (CUDA.jl) capable machine.","category":"page"},{"location":"#Usage","page":"Home","title":"Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Install GaussianSplatting.jl package:","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add https://github.com/JuliaNeuralGraphics/GaussianSplatting.jl.git","category":"page"},{"location":"#For-AMD-GPU","page":"Home","title":"For AMD GPU","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Add necessary packages: ] add AMDGPU\nRun:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using AMDGPU, GaussianSplatting\n\njulia> GaussianSplatting.gui(\"path-to-colmap-dataset-directory\"; scale=1)","category":"page"},{"location":"#For-Nvidia-GPU","page":"Home","title":"For Nvidia GPU","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Add necessary packages: ] add CUDA, cuDNN\nRun:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using CUDA, cuDNN, GaussianSplatting\n\njulia> GaussianSplatting.gui(\"path-to-colmap-dataset-directory\"; scale=1)","category":"page"},{"location":"#GPU-selection","page":"Home","title":"GPU selection","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This is required only the first time per the environment. After selecting GPU backend, restart Julia REPL.","category":"page"},{"location":"","page":"Home","title":"Home","text":"AMD GPU:\njulia> using Flux\n\njulia> Flux.gpu_backend!(\"AMDGPU\")\nNvidia GPU:\njulia> using Flux\n\njulia> Flux.gpu_backend!(\"CUDA\")","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"3D Gaussian Splatting for Real-Time Radiance Field Rendering: https://arxiv.org/abs/2308.04079\ngsplat: https://github.com/nerfstudio-project/gsplat","category":"page"}]
}