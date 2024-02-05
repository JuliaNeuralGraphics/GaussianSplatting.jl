Base.@kwdef struct OptimizationParams
    λ_dssim::Float32 = 0.2f0

    lr_points_start::Float32 = 16f-5
    lr_points_end::Float32 = 16f-7
    lr_points_steps::Int = 30_000

    lr_feature::Float32 = 25f-4
    lr_opacities::Float32 = 5f-2
    lr_scales::Float32 = 5f-3
    lr_rotations::Float32 = 1f-3

    dense_percent::Float32 = 1f-2

    densify_from_iter::Int = 500
    densify_until_iter::Int = 15000
    densification_interval::Int = 100
    densify_grad_threshold::Float32 = 2f-4

    opacity_reset_interval::Int = 3_000
end

struct SSIM{W <: Flux.Conv}
    window::W
    c1::Float32
    c2::Float32
end

function SSIM(kab; channels::Int = 3, σ::Float32 = 1.5f0, window_size::Int = 11)
    w = ImageFiltering.KernelFactors.gaussian(σ, window_size)
    w2d = reshape(reshape(w, :, 1) * reshape(w, 1, :), window_size, window_size, 1)
    window = reshape(
        repeat(w2d, 1, 1, channels),
        window_size, window_size, 1, channels)

    conv = Flux.Conv(
        (window_size, window_size), channels => channels;
        pad=(window_size ÷ 2, window_size ÷ 2),
        groups=channels, bias=false)
    copy!(conv.weight, window)

    SSIM(kab != CPU() ? Flux.gpu(conv) : conv, 0.01f0^2, 0.03f0^2)
end

# Inputs are in (W, H, C, B) format.
function (ssim::SSIM)(x::T, ref::T) where T
    μ₁, μ₂ = ssim.window(x), ssim.window(ref)
    μ₁², μ₂² = μ₁.^2, μ₂.^2
    μ₁₂ = μ₁ .* μ₂

    σ₁² = ssim.window(x.^2) .- μ₁²
    σ₂² = ssim.window(ref.^2) .- μ₂²
    σ₁₂ = ssim.window(x .* ref) .- μ₁₂

    l = ((2f0 .* μ₁₂ .+ ssim.c1) .* (2f0 .* σ₁₂ .+ ssim.c2)) ./
        ((μ₁² .+ μ₂² .+ ssim.c1) .* (σ₁² .+ σ₂² .+ ssim.c2))
    return mean(l)
end

function lr_exp_scheduler(lr_start::Float32, lr_end::Float32, steps::Int)
    function _scheduler(step::Int)
        (step < 0 || (lr_start ≈ 0f0 && lr_end ≈ 0f0)) && return 0f0

        t = clamp(Float32(step / steps), 0f0, 1f0)
        return exp(log(lr_start) * (1 - t) + log(lr_end) * t)
    end
    return _scheduler
end

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
    Trainer(
        rast, gs, dataset, optimizers, ssim,
        points_lr_scheduler, opt_params, 0, ids)
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
function get_image(trainer::Trainer, idx::Int)
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
    camera = trainer.dataset.cameras[idx]
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
        img = rast(
            means_3d, opacities, scales, rotations, shs;
            camera, sh_degree=gs.sh_degree, background)

        # From (c, w, h) to (w, h, c, 1) for SSIM.
        img_tmp = permutedims(img, (2, 3, 1))
        img_eval = reshape(img_tmp, size(img_tmp)..., 1)

        l1 = mean(abs.(img_eval .- target_image))
        s = 1f0 - ssim(img_eval, target_image)
        (1f0 - params.λ_dssim) * l1 + params.λ_dssim * s
    end

    # Apply gradients.
    for i in 1:length(θ)
        @inbounds θᵢ = θ[i]
        isempty(θᵢ) && continue

        @inbounds NU.step!(trainer.optimizers[i], θᵢ, ∇[i]; dispose=true)
    end

    if trainer.step ≤ params.densify_until_iter
        update_stats!(gs, rast.gstate.radii, rast.gstate.∇means_2d)
        do_densify =
            trainer.step ≥ params.densify_from_iter &&
            trainer.step % params.densification_interval == 0
        if do_densify
            max_screen_size::Int32 =
                trainer.step > params.opacity_reset_interval ? 20 : 0
            densify_and_prune!(
                trainer; grad_threshold=params.densify_grad_threshold,
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

function densify_and_prune!(
    trainer::Trainer; grad_threshold::Float32, min_opacity::Float32,
    max_screen_size::Int32, dense_percent::Float32,
)
    gs = trainer.gaussians
    ∇means_2d = gs.accum_∇means_2d ./ gs.denom
    ∇means_2d[isnan.(∇means_2d)] .= 0f0

    extent = trainer.dataset.camera_extent
    densify_and_clone!(trainer, ∇means_2d; grad_threshold, extent, dense_percent)
    densify_and_split!(trainer, ∇means_2d; grad_threshold, extent, dense_percent)

    prune_mask = reshape(opacity_activation.(gs.opacities) .< min_opacity, :)
    if max_screen_size > 0
        prune_mask .|= gs.max_radii .> max_screen_size
        prune_mask .|= reshape(maximum(scales_activation.(gs.scales); dims=1) .> 0.1f0 * extent, :)
    end
    prune_points!(trainer, prune_mask)
    return
end

function densify_and_clone!(
    trainer::Trainer, ∇means_2d; grad_threshold::Float32,
    extent::Float32, dense_percent::Float32,
)
    gs = trainer.gaussians

    # Extract points that satisfy the gradient condition.
    mask = ∇means_2d .≥ grad_threshold # TODO compute norm if grad is not 1D?
    mask .&= reshape(maximum(scales_activation.(gs.scales); dims=1) .≤ extent * dense_percent, :)

    new_points = gs.points[:, mask]
    new_features_dc = gs.features_dc[:, :, mask]
    new_features_rest = gs.features_rest[:, :, mask]
    new_scales = gs.scales[:, mask]
    new_rotations = gs.rotations[:, mask]
    new_opacities = gs.opacities[:, mask]

    densification_postfix!(
        trainer, new_points, new_features_dc, new_features_rest,
        new_scales, new_rotations, new_opacities)
end

function densify_and_split!(
    trainer::Trainer, ∇means_2d; grad_threshold::Float32,
    extent::Float32, dense_percent::Float32,
    n_split::Int = 2
)
    gs = trainer.gaussians
    kab = get_backend(gs)
    n = size(gs.points, 2)

    padded_grad = KA.zeros(kab, eltype(∇means_2d), n)
    padded_grad[1:length(∇means_2d)] .= ∇means_2d

    mask = padded_grad .≥ grad_threshold
    mask .&= reshape(maximum(scales_activation.(gs.scales); dims=1) .> extent * dense_percent, :)
    stds = Array(repeat(scales_activation.(gs.scales)[:, mask], 1, n_split))

    new_features_rest = repeat(gs.features_rest[:, :, mask], 1, 1, n_split)
    new_features_dc = repeat(gs.features_dc[:, :, mask], 1, 1, n_split)
    new_scales = scales_inv_activation.(stds ./ (0.8f0 * n_split))
    new_rotations = repeat(gs.rotations[:, mask], 1, n_split)
    new_opacities = repeat(gs.opacities[:, mask], 1, n_split)

    # Compute new points: sample shifts from normal distribution -> add them to new points.
    # Sampling from normal distribution on the host.
    # TODO support sampling on GPU
    rots = reinterpret(SVector{4, Float32}, reshape(Array(new_rotations), :))
    samples = Matrix{Float32}(undef, 3, length(rots))
    for (i, q) in enumerate(rots)
        δ = SVector{3, Float32}(
            rand(Normal(0f0, stds[1, i])),
            rand(Normal(0f0, stds[2, i])),
            rand(Normal(0f0, stds[3, i])))
        samples[:, i] .= quat2mat(normalize(q)) * δ # TODO transpose rot?
    end
    new_points = adapt(kab, samples) .+ repeat(gs.points[:, mask], 1, n_split)

    densification_postfix!(
        trainer, new_points, new_features_dc, new_features_rest,
        new_scales, new_rotations, new_opacities)

    prune_mask = vcat(mask, KA.zeros(kab, Bool, n_split * sum(mask)))
    prune_points!(trainer, prune_mask)
    return
end

function prune_points!(trainer::Trainer, prune_mask)
    gs = trainer.gaussians
    opts = trainer.optimizers

    valid_mask = .!prune_mask

    _prune_optimizer!(opts.points, valid_mask, gs.points)
    gs.points = gs.points[:, valid_mask]

    _prune_optimizer!(opts.features_dc, valid_mask, gs.features_dc)
    gs.features_dc = gs.features_dc[:, :, valid_mask]

    _prune_optimizer!(opts.features_rest, valid_mask, gs.features_rest)
    gs.features_rest = gs.features_rest[:, :, valid_mask]

    _prune_optimizer!(opts.scales, valid_mask, gs.scales)
    gs.scales = gs.scales[:, valid_mask]

    _prune_optimizer!(opts.rotations, valid_mask, gs.rotations)
    gs.rotations = gs.rotations[:, valid_mask]

    _prune_optimizer!(opts.opacities, valid_mask, gs.opacities)
    gs.opacities = gs.opacities[:, valid_mask]

    gs.max_radii = gs.max_radii[valid_mask]
    gs.accum_∇means_2d = gs.accum_∇means_2d[valid_mask]
    gs.denom = gs.denom[valid_mask]
    return
end

function densification_postfix!(
    trainer::Trainer, new_points, new_features_dc, new_features_rest,
    new_scales, new_rotations, new_opacities,
)
    gs = trainer.gaussians

    # TODO unsafe free

    gs.points = cat(gs.points, new_points; dims=ndims(new_points))
    _append_optimizer!(trainer.optimizers.points, new_points)

    gs.features_dc = cat(gs.features_dc, new_features_dc; dims=ndims(new_features_dc))
    _append_optimizer!(trainer.optimizers.features_dc, new_features_dc)

    gs.features_rest = cat(gs.features_rest, new_features_rest; dims=ndims(new_features_rest))
    _append_optimizer!(trainer.optimizers.features_rest, new_features_rest)

    gs.scales = cat(gs.scales, new_scales; dims=ndims(new_scales))
    _append_optimizer!(trainer.optimizers.scales, new_scales)

    gs.rotations = cat(gs.rotations, new_rotations; dims=ndims(new_rotations))
    _append_optimizer!(trainer.optimizers.rotations, new_rotations)

    gs.opacities = cat(gs.opacities, new_opacities; dims=ndims(new_opacities))
    _append_optimizer!(trainer.optimizers.opacities, new_opacities)

    kab = get_backend(gs)
    n = size(gs.points, 2)
    gs.max_radii = KA.zeros(kab, Int32, n)
    gs.accum_∇means_2d = KA.zeros(kab, Float32, n)
    gs.denom = KA.zeros(kab, Float32, n)
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
