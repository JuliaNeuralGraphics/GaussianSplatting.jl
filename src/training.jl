# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
mutable struct Trainer{
    R <: GaussianRasterizer,
    G <: GaussianModel,
    D <: ColmapDataset,
    C <: GPUArrays.AllocCache,
    F,
    O,
}
    rast::R
    gaussians::G
    dataset::D
    optimizers::O

    cache::C

    points_lr_scheduler::F
    opt_params::OptimizationParams

    densify::Bool
    step::Int
    ids::Vector{Int}

    # Per train camera depth anchors; empty when depth supervision is off.
    depth_anchors::Vector{Maybe{DepthAnchor}}
end

function Trainer(
    rast::GaussianRasterizer, gs::GaussianModel,
    dataset::ColmapDataset, opt_params::OptimizationParams;
)
    ϵ = 1f-15
    kab = get_backend(gs)
    cache = GPUArrays.AllocCache()

    optimizers = (;
        points=NU.Adam(kab, gs.points; lr=opt_params.lr_points_start * dataset.camera_extent, ϵ),
        features_dc=NU.Adam(kab, gs.features_dc; lr=opt_params.lr_feature, ϵ),
        features_rest=NU.Adam(kab, gs.features_rest; lr=opt_params.lr_feature / 20f0, ϵ),
        opacities=NU.Adam(kab, gs.opacities; lr=opt_params.lr_opacities, ϵ),
        scales=NU.Adam(kab, gs.scales; lr=opt_params.lr_scales, ϵ),
        rotations=NU.Adam(kab, gs.rotations; lr=opt_params.lr_rotations, ϵ))

    points_lr_scheduler = lr_exp_scheduler(
        opt_params.lr_points_start * dataset.camera_extent,
        opt_params.lr_points_end * dataset.camera_extent,
        opt_params.lr_points_steps)

    ids = collect(1:length(dataset))
    densify = true
    step = 0

    depth_anchors = opt_params.use_depth_loss ?
        setup_depth_supervision(rast, dataset, opt_params) :
        Maybe{DepthAnchor}[]

    Trainer(
        rast, gs, dataset, optimizers, cache,
        points_lr_scheduler, opt_params, densify, step, ids,
        depth_anchors)
end

function setup_depth_supervision(
    rast::GaussianRasterizer, dataset::ColmapDataset,
    opt_params::OptimizationParams,
)
    disabled = Maybe{DepthAnchor}[]
    if rast.mode != :rgbd
        @warn "Depth supervision requires a `:rgbd` rasterizer, disabling."
        return disabled
    end
    if !any(!isnothing, dataset.train_depths)
        @warn "Depth supervision enabled, but no depth priors were found, disabling."
        return disabled
    end
    points = Array(dataset.points)
    if isempty(points)
        @warn "Depth supervision requires an init point cloud to fit anchors, disabling."
        return disabled
    end
    return fit_depth_anchors(
        points, dataset.train_cameras, dataset.train_depths;
        mode=opt_params.depth_loss_mode)
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

        eval_ssim += mean(fused_ssim(image_eval; ref=target_image))
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

    anchor = isempty(trainer.depth_anchors) ? nothing : trainer.depth_anchors[idx]

    kab = get_backend(rast)
    GPUArrays.@cached trainer.cache begin
        # Depth supervision target for this view (constant w.r.t. AD).
        depth_data = if anchor ≡ nothing
            nothing
        else
            prior = adapt(kab, trainer.dataset.train_depths[idx])
            target, half_band, valid = depth_target(
                anchor, prior, trainer.dataset.train_depth_qsteps[idx])
            # Depth dominates early geometry formation;
            # photometric loss wins fine detail late.
            decay = DEPTH_LOSS_FINAL_SCALE^clamp(
                Float32(trainer.step / params.depth_loss_steps), 0f0, 1f0)
            weight = params.depth_loss_weight * decay
            (; target, half_band, valid, weight)
        end

        loss, ∇ = Zygote.withgradient(
            θ...,
        ) do means_3d, features_dc, features_rest, opacities, scales, rotations
            image_features = rast(
                means_3d, opacities, scales, rotations, features_dc, features_rest;
                camera, sh_degree=gs.sh_degree, background)

            # Unconditional slice (a no-op for `:rgb`): a branch whose
            # else-arm aliases `image_features` unsliced makes Zygote
            # mis-route the gradient of the alias past the `getindex`
            # pullback once the depth term adds a second use, crashing
            # gradient accumulation with a shape mismatch.
            image = image_features[1:3, :, :]

            # From (c, w, h) to (w, h, c, 1) for SSIM.
            image_tmp = permutedims(image, (2, 3, 1))
            image_eval = reshape(image_tmp, size(image_tmp)..., 1)

            l1 = mean(abs.(image_eval .- target_image))
            s = 1f0 - mean(fused_ssim(image_eval; ref=target_image))
            total = (1f0 - params.λ_dssim) * l1 + params.λ_dssim * s

            if depth_data ≢ nothing
                depth_img = image_features[4, :, :]
                total += depth_data.weight * ssi_depth_loss(depth_img;
                    transmittance=rast.istate.accum_α,
                    depth_data.target, depth_data.half_band, depth_data.valid,
                    depth_floor=anchor.floor)
            end
            total
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
                pruning_extent=trainer.dataset.camera_extent,
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
