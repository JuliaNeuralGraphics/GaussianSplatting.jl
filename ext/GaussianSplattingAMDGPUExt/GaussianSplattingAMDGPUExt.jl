module GaussianSplattingAMDGPUExt

using Adapt
using AMDGPU
using KernelAbstractions
using GaussianSplatting
using PrecompileTools

@setup_workload let
    kab = GaussianSplatting.gpu_backend()

    # TODO KernelAbstractions.functional(kab)
    (kab isa ROCBackend && AMDGPU.functional()) || return

    @info "Precompiling for `$kab` GPU backend."

    points = adapt(kab, rand(Float32, 3, 128))
    colors = adapt(kab, rand(Float32, 3, 128))
    scales = adapt(kab, rand(Float32, 3, 128))

    camera = GaussianSplatting.Camera(; fx=100f0, fy=100f0, width=256, height=256)

    opt_params = GaussianSplatting.OptimizationParams()
    gaussians = GaussianSplatting.GaussianModel(points, colors, scales; max_sh_degree=0)
    rasterizer = GaussianSplatting.GaussianRasterizer(kab, camera; auxiliary=false)

    @compile_workload begin
        rasterizer(
            gaussians.points, gaussians.opacities, gaussians.scales,
            gaussians.rotations, gaussians.features_dc;
            camera, sh_degree=gaussians.sh_degree, covisibility=nothing)
    end
    @info "Done precompiling!"
    return
end

end
