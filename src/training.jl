# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
mutable struct Trainer{
    R <: GaussianRasterizer,
    G <: GaussianModel,
    D <: ColmapDataset,
    S <: SSIM,
    C <: GPUArrays.AllocCache,
    F,
    O,
}
    rast::R
    gaussians::G
    dataset::D
    optimizers::O
    ssim::S

    cache::C

    points_lr_scheduler::F
    opt_params::OptimizationParams

    densify::Bool
    step::Int
    ids::Vector{Int}
end

function Trainer(
    rast::GaussianRasterizer, gs::GaussianModel,
    dataset::ColmapDataset, opt_params::OptimizationParams;
)
    ϵ = 1f-15
    kab = get_backend(gs)
    camera_extent = min(dataset.camera_extent, 4f0) # TODO squeeze scene into unit box
    cache = GPUArrays.AllocCache()

    optimizers = (;
        points=NU.Adam(kab, gs.points; lr=opt_params.lr_points_start * camera_extent, ϵ),
        features_dc=NU.Adam(kab, gs.features_dc; lr=opt_params.lr_feature, ϵ),
        features_rest=NU.Adam(kab, gs.features_rest; lr=opt_params.lr_feature / 20f0, ϵ),
        opacities=NU.Adam(kab, gs.opacities; lr=opt_params.lr_opacities, ϵ),
        scales=NU.Adam(kab, gs.scales; lr=opt_params.lr_scales, ϵ),
        rotations=NU.Adam(kab, gs.rotations; lr=opt_params.lr_rotations, ϵ))
    ssim = SSIM(kab)

    points_lr_scheduler = lr_exp_scheduler(
        opt_params.lr_points_start * camera_extent,
        opt_params.lr_points_end * camera_extent,
        opt_params.lr_points_steps)

    ids = collect(1:length(dataset))
    densify = true
    step = 0
    Trainer(
        rast, gs, dataset, optimizers, ssim, cache,
        points_lr_scheduler, opt_params, densify, step, ids)
end

function bson_params(opt::NU.Adam)
    (;
        μ=[adapt(CPU(), i) for i in opt.μ],
        ν=[adapt(CPU(), i) for i in opt.ν],
        current_step=opt.current_step)
end

function set_from_bson!(opt::NU.Adam, θ)
    kab = get_backend(opt)
    opt.μ = [adapt(kab, i) for i in θ.μ]
    opt.ν = [adapt(kab, i) for i in θ.ν]
    opt.current_step = θ.current_step
    return
end

function save_state(trainer::Trainer, filename::String)
    optimizers = (;
        points=bson_params(trainer.optimizers.points),
        features_dc=bson_params(trainer.optimizers.features_dc),
        features_rest=bson_params(trainer.optimizers.features_rest),
        opacities=bson_params(trainer.optimizers.opacities),
        scales=bson_params(trainer.optimizers.scales),
        rotations=bson_params(trainer.optimizers.rotations))

    camera = trainer.dataset.train_cameras[1]
    BSON.bson(filename, Dict(
        :gaussians => bson_params(trainer.gaussians),
        :optimizers => optimizers,
        :step => trainer.step,
        :camera => camera,
    ))
    return
end

function load_state!(trainer::Trainer, filename::String)
    θ = BSON.load(filename)
    optimizers = θ[:optimizers]
    set_from_bson!(trainer.gaussians, θ[:gaussians])

    set_from_bson!(trainer.optimizers.points, optimizers.points)
    set_from_bson!(trainer.optimizers.features_dc, optimizers.features_dc)
    set_from_bson!(trainer.optimizers.features_rest, optimizers.features_rest)
    set_from_bson!(trainer.optimizers.opacities, optimizers.opacities)
    set_from_bson!(trainer.optimizers.scales, optimizers.scales)
    set_from_bson!(trainer.optimizers.rotations, optimizers.rotations)

    trainer.step = θ[:step]
    return
end

function reset_opacity!(trainer::Trainer)
    reset_opacity!(trainer.gaussians)
    NU.reset!(trainer.optimizers.opacities)
end

# Convert image from UInt8 to Float32 & permute from (c, w, h) to (w, h, c, 1).
function get_image(trainer::Trainer, idx::Integer, set::Symbol)
    kab = get_backend(trainer.gaussians)
    target_image = get_image(trainer.dataset, kab, idx, set)
    target_image = permutedims(target_image, (2, 3, 1))
    return reshape(target_image, size(target_image)..., 1)
end

function validate(trainer::Trainer)
    gs = trainer.gaussians
    rast = trainer.rast
    ssim = trainer.ssim
    dataset = trainer.dataset

    eval_ssim = 0f0
    eval_mse = 0f0
    eval_psnr = 0f0
    for (idx, camera) in enumerate(dataset.test_cameras)
        target_image = get_image(trainer, idx, :test)

        image_features = rast(
            gs.points, gs.opacities, gs.scales,
            gs.rotations, gs.features_dc, gs.features_rest;
            camera, sh_degree=gs.sh_degree)

        image = if rast.mode == :rgbd
            image_features[1:3, :, :]
        else
            image_features
        end

        # From (c, w, h) to (w, h, c, 1) for SSIM.
        image_tmp = permutedims(image, (2, 3, 1))
        image_eval = reshape(image_tmp, size(image_tmp)..., 1)

        eval_ssim += ssim(image_eval, target_image)
        eval_mse += mse(image_eval, target_image)
        eval_psnr += psnr(image_eval, target_image)
    end
    eval_ssim /= length(dataset.test_cameras)
    eval_mse /= length(dataset.test_cameras)
    eval_psnr /= length(dataset.test_cameras)
    return (; eval_ssim, eval_mse, eval_psnr)
end

function step!(trainer::Trainer)
    trainer.step += 1
    update_lr!(trainer)

    gs = trainer.gaussians
    rast = trainer.rast
    ssim = trainer.ssim
    params = trainer.opt_params

    if trainer.step % 1000 == 0 && gs.sh_degree < gs.max_sh_degree
        gs.sh_degree += 1
    end

    if (trainer.step - 1) % length(trainer.dataset) == 0
        shuffle!(trainer.ids)
    end
    idx = trainer.ids[(trainer.step - 1) % length(trainer.dataset) + 1]
    camera = trainer.dataset.train_cameras[idx]
    target_image = get_image(trainer, idx, :train)
    background = rand(SVector{3, Float32})

    θ = (
        gs.points, gs.features_dc, gs.features_rest,
        gs.opacities, gs.scales, gs.rotations)

    kab = get_backend(rast)
    GPUArrays.@cached trainer.cache begin
        loss, ∇ = Zygote.withgradient(
            θ...,
        ) do means_3d, features_dc, features_rest, opacities, scales, rotations
            image_features = rast(
                means_3d, opacities, scales, rotations, features_dc, features_rest;
                camera, sh_degree=gs.sh_degree, background)

            image = if rast.mode == :rgbd
                image_features[1:3, :, :]
            else
                image_features
            end

            # From (c, w, h) to (w, h, c, 1) for SSIM.
            image_tmp = permutedims(image, (2, 3, 1))
            image_eval = reshape(image_tmp, size(image_tmp)..., 1)

            l1 = mean(abs.(image_eval .- target_image))
            s = 1f0 - ssim(image_eval, target_image)
            (1f0 - params.λ_dssim) * l1 + params.λ_dssim * s
        end

        # Apply gradients.
        for i in 1:length(θ)
            θᵢ = θ[i]
            isempty(θᵢ) && continue
            NU.step!(trainer.optimizers[i], θᵢ, ∇[i]; dispose=false)
        end
    end

    if trainer.densify && trainer.step ≤ params.densify_until_iter
        update_stats!(gs, rast.gstate.radii,
            rast.gstate.∇means_2d, camera.intrinsics.resolution)
        do_densify =
            trainer.step ≥ params.densify_from_iter &&
            trainer.step % params.densification_interval == 0
        if do_densify
            GPUArrays.unsafe_free!(trainer.cache)

            max_screen_size::Int32 =
                trainer.step > params.opacity_reset_interval ? 20 : 0
            densify_and_prune!(gs, trainer.optimizers;
                extent=trainer.dataset.camera_extent,
                grad_threshold=params.densify_grad_threshold,
                min_opacity=0.05f0, max_screen_size, params.dense_percent)
        end

        if trainer.step % params.opacity_reset_interval == 0
            reset_opacity!(trainer)
        end
    end
    return loss
end

function update_lr!(trainer::Trainer)
    trainer.optimizers.points.lr = trainer.points_lr_scheduler(trainer.step)
    return
end
