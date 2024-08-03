module GaussianSplattingAMDGPUExt

using Adapt
using AMDGPU
using KernelAbstractions
using GaussianSplatting
using PrecompileTools
using Statistics
using Zygote

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
    ssim = GaussianSplatting.SSIM(kab)

    θ = (
        gaussians.points, gaussians.features_dc, gaussians.features_rest,
        gaussians.opacities, gaussians.scales, gaussians.rotations)
    target_image = adapt(kab, rand(Float32, 256, 256, 3, 1))

    @compile_workload begin
        Zygote.gradient(
            θ...,
        ) do means_3d, features_dc, features_rest, opacities, scales, rotations
            shs = isempty(features_rest) ?
                features_dc : hcat(features_dc, features_rest)
            img = rasterizer(
                means_3d, opacities, scales, rotations, shs;
                camera, sh_degree=gaussians.sh_degree)

            # From (c, w, h) to (w, h, c, 1) for SSIM.
            img_tmp = permutedims(img, (2, 3, 1))
            img_eval = reshape(img_tmp, size(img_tmp)..., 1)

            l1 = mean(abs.(img_eval .- target_image))
            s = 1f0 - ssim(img_eval, target_image)
            (1f0 - opt_params.λ_dssim) * l1 + opt_params.λ_dssim * s
        end
    end
    @info "Done precompiling!"
    return
end

end
