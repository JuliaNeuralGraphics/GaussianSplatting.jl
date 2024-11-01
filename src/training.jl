# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
mutable struct Trainer{
    R <: GaussianRasterizer,
    G <: GaussianModel,
    D <: ColmapDataset,
    S <: SSIM,
    F,
    O,
}
    rast::R
    gaussians::G
    dataset::D
    optimizers::O
    ssim::S

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
    camera_extent = dataset.camera_extent

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
        rast, gs, dataset, optimizers, ssim,
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

    BSON.bson(filename, Dict(
        :gaussians => bson_params(trainer.gaussians),
        :optimizers => optimizers,
        :step => trainer.step,
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
function get_image(trainer::Trainer, idx::Integer)
    kab = get_backend(trainer.gaussians)
    target_image = get_image(trainer.dataset, kab, idx)
    target_image = permutedims(target_image, (2, 3, 1))
    return reshape(target_image, size(target_image)..., 1)
end

function step!(trainer::Trainer)
    trainer.step += 1
    update_lr!(trainer)

    # Aliases.
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
    # camera = trainer.dataset.cameras[idx]
    camera = trainer.dataset.cameras[1]
    target_image = get_image(trainer, idx)
    background = rand(SVector{3, Float32})

    θ = (
        gs.points, gs.features_dc, gs.features_rest,
        gs.opacities, gs.scales, gs.rotations)

    loss, ∇ = Zygote.withgradient(
        θ...,
    ) do means_3d, features_dc, features_rest, opacities, scales, rotations
        shs = isempty(features_rest) ?
            features_dc : hcat(features_dc, features_rest)
        # image = rast(Val{:alloc}(),
        image = rast(
            means_3d, opacities, scales, rotations, shs;
            camera, sh_degree=gs.sh_degree, background)

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

        NU.step!(trainer.optimizers[i], θᵢ, ∇[i]; dispose=true)
    end

    if trainer.densify && trainer.step ≤ params.densify_until_iter
        update_stats!(gs, rast.gstate.radii,
            rast.gstate.∇means_2d, camera.intrinsics.resolution)
        do_densify =
            trainer.step ≥ params.densify_from_iter &&
            trainer.step % params.densification_interval == 0
        if do_densify
            max_screen_size::Int32 =
                trainer.step > params.opacity_reset_interval ? 20 : 0
            densify_and_prune!(gs, trainer.optimizers;
                extent=trainer.dataset.camera_extent,
                grad_threshold=params.densify_grad_threshold,
                min_opacity=0.05f0, max_screen_size, params.dense_percent)
        end

        if trainer.step % params.opacity_reset_interval == 0 # TODO or if white background
            reset_opacity!(trainer)
        end
    end

    return loss
end

function update_lr!(trainer::Trainer)
    trainer.optimizers.points.lr = trainer.points_lr_scheduler(trainer.step)
    return
end

# Densification & pruning.

function densify_and_prune!(gs::GaussianModel, optimizers;
    extent::Float32,
    grad_threshold::Float32, min_opacity::Float32,
    max_screen_size::Int32, dense_percent::Float32,
)
    ∇means_2d = gs.accum_∇means_2d ./ gs.denom
    ∇means_2d[isnan.(∇means_2d)] .= 0f0

    densify_clone!(gs, optimizers; ∇means_2d, grad_threshold, extent, dense_percent)
    densify_split!(gs, optimizers; ∇means_2d, grad_threshold, extent, dense_percent)

    # Prune points that are too transparent, occupy too much space in image space
    # and have high scale in world space.
    valid_mask = reshape(opacity_activation.(gs.opacities) .> min_opacity, :)
    if max_screen_size > 0
        γ = 0.1f0 * extent
        valid_mask .&=
            (gs.max_radii .< max_screen_size) .&&
            reshape(maximum(scales_activation.(gs.scales); dims=1) .< γ, :)
    end
    prune_points!(gs, optimizers, valid_mask)
    return
end

function densify_clone!(gs::GaussianModel, optimizers;
    ∇means_2d, grad_threshold::Float32,
    extent::Float32, dense_percent::Float32,
)
    # Clone gaussians that have high gradient and small size.
    γ = extent * dense_percent
    mask =
        ∇means_2d .> grad_threshold .&&
        reshape(maximum(scales_activation.(gs.scales); dims=1) .< γ, :)

    new_points = gs.points[:, mask]
    new_features_dc = gs.features_dc[:, :, mask]
    new_features_rest = isempty(gs.features_rest) ?
        gs.features_rest : gs.features_rest[:, :, mask]
    new_scales = gs.scales[:, mask]
    new_rotations = gs.rotations[:, mask]
    new_opacities = gs.opacities[:, mask]

    new_ids = gs.ids ≡ nothing ? nothing : gs.ids[mask]

    densification_postfix!(gs, optimizers;
        new_points, new_features_dc, new_features_rest,
        new_scales, new_rotations, new_opacities, new_ids)
end

function densify_split!(gs::GaussianModel, optimizers;
    ∇means_2d, grad_threshold::Float32,
    extent::Float32, dense_percent::Float32,
)
    kab = get_backend(gs)
    n = size(gs.points, 2)
    n_split = 2

    padded_grad = KA.zeros(kab, eltype(∇means_2d), n)
    padded_grad[1:length(∇means_2d)] .= ∇means_2d

    # Split gaussians that have high gradient and big size.
    γ = extent * dense_percent
    mask =
        padded_grad .≥ grad_threshold .&&
        reshape(maximum(scales_activation.(gs.scales); dims=1) .> γ, :)
    stds = repeat(scales_activation.(gs.scales)[:, mask], 1, n_split)

    new_features_rest = isempty(gs.features_rest) ?
        gs.features_rest : repeat(gs.features_rest[:, :, mask], 1, 1, n_split)
    new_features_dc = repeat(gs.features_dc[:, :, mask], 1, 1, n_split)
    new_scales = scales_inv_activation.(stds ./ (0.8f0 * n_split))
    new_rotations = repeat(gs.rotations[:, mask], 1, n_split)
    new_opacities = repeat(gs.opacities[:, mask], 1, n_split)

    new_ids = gs.ids ≡ nothing ? nothing : repeat(gs.ids[mask], n_split)

    new_points = repeat(gs.points[:, mask], 1, n_split)
    n_new_points = size(new_points, 2)
    if n_new_points > 0
        _add_split_noise!(kab)(
            reinterpret(SVector{3, Float32}, new_points),
            reinterpret(SVector{4, Float32}, new_rotations),
            reinterpret(SVector{3, Float32}, stds); ndrange=n_new_points)
    end

    densification_postfix!(gs, optimizers;
        new_points, new_features_dc, new_features_rest,
        new_scales, new_rotations, new_opacities, new_ids)

    # Prune gaussians that have small gradient or small size
    # ignoring newly inserted gaussians.
    valid_mask = vcat(.!mask, KA.ones(kab, Bool, n_new_points))
    prune_points!(gs, optimizers, valid_mask)
    return
end

@kernel cpu=false inbounds=true function _add_split_noise!(points, @Const(rots), @Const(stds))
    i = @index(Global)
    σ = stds[i]
    ξ = SVector{3, Float32}(
        randn(Float32) * σ[1],
        randn(Float32) * σ[2],
        randn(Float32) * σ[3])

    q = rots[i]
    R = unnorm_quat2rot(q)
    p = points[i]
    points[i] = p .+ R * ξ
end

function prune_points!(gs::GaussianModel, optimizers, valid_mask)
    _prune_optimizer!(optimizers.points, valid_mask, gs.points)
    gs.points = gs.points[:, valid_mask]

    _prune_optimizer!(optimizers.features_dc, valid_mask, gs.features_dc)
    gs.features_dc = gs.features_dc[:, :, valid_mask]

    if !isempty(gs.features_rest)
        _prune_optimizer!(optimizers.features_rest, valid_mask, gs.features_rest)
        gs.features_rest = gs.features_rest[:, :, valid_mask]
    end

    _prune_optimizer!(optimizers.scales, valid_mask, gs.scales)
    gs.scales = gs.scales[:, valid_mask]

    _prune_optimizer!(optimizers.rotations, valid_mask, gs.rotations)
    gs.rotations = gs.rotations[:, valid_mask]

    _prune_optimizer!(optimizers.opacities, valid_mask, gs.opacities)
    gs.opacities = gs.opacities[:, valid_mask]

    gs.max_radii = gs.max_radii[valid_mask]
    gs.accum_∇means_2d = gs.accum_∇means_2d[valid_mask]
    gs.denom = gs.denom[valid_mask]
    if gs.ids ≢ nothing
        gs.ids = gs.ids[valid_mask]
    end
    return
end

function densification_postfix!(
    gs::GaussianModel, optimizers;
    new_points, new_features_dc, new_features_rest,
    new_scales, new_rotations, new_opacities, new_ids,
)
    gs.points = cat(gs.points, new_points; dims=ndims(new_points))
    _append_optimizer!(optimizers.points, new_points)

    gs.features_dc = cat(gs.features_dc, new_features_dc; dims=ndims(new_features_dc))
    _append_optimizer!(optimizers.features_dc, new_features_dc)

    if !isempty(gs.features_rest)
        gs.features_rest = cat(gs.features_rest, new_features_rest; dims=ndims(new_features_rest))
        _append_optimizer!(optimizers.features_rest, new_features_rest)
    end

    gs.scales = cat(gs.scales, new_scales; dims=ndims(new_scales))
    _append_optimizer!(optimizers.scales, new_scales)

    gs.rotations = cat(gs.rotations, new_rotations; dims=ndims(new_rotations))
    _append_optimizer!(optimizers.rotations, new_rotations)

    gs.opacities = cat(gs.opacities, new_opacities; dims=ndims(new_opacities))
    _append_optimizer!(optimizers.opacities, new_opacities)

    kab = get_backend(gs)
    n = size(gs.points, 2)
    gs.max_radii = KA.zeros(kab, Int32, n)
    gs.accum_∇means_2d = KA.zeros(kab, Float32, n)
    gs.denom = KA.zeros(kab, Float32, n)

    if gs.ids ≢ nothing
        gs.ids = cat(gs.ids, new_ids; dims=1)
    end
    return
end

function _append_optimizer!(opt::NU.Adam, extension)
    kab = get_backend(extension)
    dims = ndims(extension)

    μ = opt.μ[1]
    μ̂ = KA.zeros(kab, eltype(μ), size(extension))
    μ = cat(reshape(μ, size(extension)[1:end - 1]..., :), μ̂; dims)
    opt.μ[1] = reshape(μ, :)

    ν = opt.ν[1]
    ν̂ = KA.zeros(kab, eltype(ν), size(extension))
    ν = cat(reshape(ν, size(extension)[1:end - 1]..., :), ν̂; dims)
    opt.ν[1] = reshape(ν, :)
    return
end

function _prune_optimizer!(opt::NU.Adam, mask, x)
    d = ntuple(i -> Colon(), ndims(x) - 1)
    opt.μ[1] = reshape(reshape(opt.μ[1], size(x))[d..., mask], :)
    opt.ν[1] = reshape(reshape(opt.ν[1], size(x))[d..., mask], :)
    return
end
