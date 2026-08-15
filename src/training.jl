# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

"""
Per-term breakdown of the training loss, refreshed by every [`step!`](@ref).

Terms are stored **as weighted**, i.e. exactly the contribution each makes to
`total`, so they sum to it. That is the form that answers what the breakdown
exists for — which term is actually driving the optimizer — and it makes a
mis-scaled weight visible directly rather than requiring mental arithmetic
against `OptimizationParams`. Divide by the corresponding weight for the raw
term value.

A term sitting at exactly `0` is one that did not run this step: disabled, or
still behind its warm-up iteration (`normal_from_iter`, `sky_loss_from_iter`).
"""
Base.@kwdef mutable struct LossBreakdown
    total::Float32 = 0f0
    l1::Float32 = 0f0
    ssim::Float32 = 0f0
    # `regularization_loss` of the active strategy: `0` for `DefaultStrategy`,
    # opacity + scale L1 for `MCMCStrategy`.
    reg::Float32 = 0f0
    tv::Float32 = 0f0
    depth::Float32 = 0f0
    sky::Float32 = 0f0
    mask::Float32 = 0f0
    flatten::Float32 = 0f0
    normal::Float32 = 0f0
end

# Every field, in the order terms are reported.
const LOSS_TERMS = (:l1, :ssim, :reg, :tv, :depth, :sky, :mask, :flatten, :normal)
const LOSS_FIELDS = (:total, LOSS_TERMS...)

# How often the GUI's train loop reports the breakdown to the console.
# `main` reports on its own validation cadence.
const LOSS_REPORT_INTERVAL = 100

# Steps the moving average covers. A few report intervals, so consecutive
# reports overlap and a trend is visible rather than being resampled.
const LOSS_EMA_HORIZON = 200

# Steps between samples of the per-term history the GUI plots, and how many
# samples it keeps.
const LOSS_HISTORY_INTERVAL = 10
const LOSS_HISTORY_CAPACITY = 512

"""
Per-term loss curves for the whole run: what the `Training Details` plot draws.

Sampled every `interval` steps; once the run outgrows `LOSS_HISTORY_CAPACITY`
samples, every second one is dropped and the interval doubles. So the curves
always span the whole run at bounded cost, rather than a window scrolling over
its tail — a training curve is read against where it started.

Values are the bias-corrected moving averages, not the per-step breakdowns:
each step scores a different view, and at one sample per 10 steps the raw
values would be an unreadable band of view difficulty.
"""
mutable struct LossHistory
    steps::Vector{Float32}
    terms::NamedTuple{LOSS_FIELDS, NTuple{length(LOSS_FIELDS), Vector{Float32}}}
    interval::Int
    # Bumped on every sample, so the GUI can publish a snapshot to its UI
    # thread only when there is something new to plot (see `RenderWorker`).
    version::Int
end

LossHistory() = LossHistory(
    Float32[],
    NamedTuple{LOSS_FIELDS}(ntuple(_ -> Float32[], length(LOSS_FIELDS))),
    LOSS_HISTORY_INTERVAL, 0)

# Halve the sample count & double the interval, keeping every second sample.
function thin!(history::LossHistory)
    history.steps = history.steps[1:2:end]
    history.terms = map(values -> values[1:2:end], history.terms)
    history.interval *= 2
    return history
end

"""
Copy of the curves, for a thread that does not own the history: the trainer
keeps appending to (and reallocating) its own vectors.
"""
snapshot(history::LossHistory) = (;
    steps=copy(history.steps),
    terms=map(copy, history.terms),
    version=history.version)

"""
The step's breakdown plus an exponential moving average of it.

Every step scores a *different randomly sampled view*, so consecutive
breakdowns differ by view difficulty as much as by optimization progress —
enough that raw values a hundred steps apart cannot be read as a trend. The
average is what makes a term's direction legible.
"""
mutable struct LossLog
    current::LossBreakdown
    # Raw EMA accumulator: biased low early, since it starts from zero.
    # `smoothed` applies the correction rather than seeding, so a term that
    # switches on mid-run (`normal_from_iter`) still ramps honestly from zero
    # instead of jumping to its first observed value.
    ema::LossBreakdown
    α::Float32
    n::Int
    # Sampled averages over the run, for the GUI's plot.
    history::LossHistory
end

LossLog(; horizon::Int = LOSS_EMA_HORIZON) = LossLog(
    LossBreakdown(), LossBreakdown(), 1f0 / Float32(horizon), 0, LossHistory())

function update_ema!(log::LossLog)
    log.n += 1
    for name in LOSS_FIELDS
        prev = getfield(log.ema, name)
        setfield!(log.ema, name,
            prev + log.α * (getfield(log.current, name) - prev))
    end
    return log
end

"""
The `1 - (1 - α)^n` de-biasing factor.
"""
bias_correction(log::LossLog) = 1f0 - (1f0 - log.α)^log.n

"""
Bias-corrected moving average, as a `LossBreakdown`.
"""
function smoothed(log::LossLog)
    correction = bias_correction(log)
    correction > 0f0 || return LossBreakdown()

    b = LossBreakdown()
    for name in LOSS_FIELDS
        setfield!(b, name, getfield(log.ema, name) / correction)
    end
    return b
end

"""
Append the current averages to `history`, if `step` lands on a sampling point.
`step` is the trainer's own step, so a resumed checkpoint's curves continue
from where the run left off rather than restarting at zero.

Defined here rather than beside `LossHistory`: it needs `smoothed`.
"""
function record!(history::LossHistory, log::LossLog, step::Int)
    # Measured from the last sample rather than as `step % interval`: after a
    # `thin!` the kept samples sit on the old grid, so a modulo rule would leave
    # one short gap at every doubling.
    isempty(history.steps) ||
        step - history.steps[end] ≥ history.interval || return history

    averages = smoothed(log)
    push!(history.steps, Float32(step))
    for name in LOSS_FIELDS
        push!(getfield(history.terms, name), getfield(averages, name))
    end
    history.version += 1

    length(history.steps) > LOSS_HISTORY_CAPACITY && thin!(history)
    return history
end

function smoothed_total(log::LossLog)
    correction = bias_correction(log)
    correction > 0f0 || return 0f0
    return log.ema.total / correction
end

function format_breakdown(b::LossBreakdown; digits::Int = 5)
    io = IOBuffer()
    for name in LOSS_TERMS
        v = getfield(b, name)
        iszero(v) && continue
        position(io) == 0 || print(io, " ")
        print(io, name, "=", round(v; digits))
    end
    return String(take!(io))
end

mutable struct Trainer{
    R <: GaussianRasterizer,
    G <: GaussianModel,
    D <: ColmapDataset,
    C <: GPUArrays.AllocCache,
    S <: AbstractStrategy,
    F,
    O,
}
    rast::R
    gaussians::G
    dataset::D
    # Reads the views the train loop is about to want, ahead of it (`dataset.jl`).
    loader::ViewLoader
    optimizers::O

    cache::C

    points_lr_scheduler::F
    opt_params::OptimizationParams

    strategy::S
    densify::Bool
    step::Int
    ids::Vector{Int}

    # Per train camera depth anchors; empty when depth supervision is off.
    depth_anchors::Vector{Maybe{DepthAnchor}}
    # Per train camera appearance grids; `nothing` when disabled.
    bilateral_grid::Maybe{BilateralGrid}
    # Far-field shell composited behind the scene; `nothing` when disabled.
    sky::Maybe{SkyDome}
    # Whether sky-mask supervision is active (requires masks & a sky dome).
    sky_loss::Bool
    # Whether geometry regularization is active (see `geometry_regularization.jl`).
    normals::Bool
    # Whether coverage masking is active (see `masking.jl`).
    masks::Bool

    # Per-term loss contributions & their moving average (see `LossLog`).
    losses::LossLog
end

function Trainer(
    rast::GaussianRasterizer, gs::GaussianModel,
    dataset::ColmapDataset, opt_params::OptimizationParams;
    strategy::AbstractStrategy = DefaultStrategy(gs),
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

    depth_anchors = if opt_params.use_depth_loss && dataset.has_depth_priors
        @info "Use depth priors for supervision."
        setup_depth_supervision(rast, dataset, opt_params)
    else
        Maybe{DepthAnchor}[]
    end

    bilateral_grid = if opt_params.use_bilateral_grid
        @info "Using bilateral grid of `$(opt_params.bilateral_grid_size)` size."
        BilateralGrid(kab, length(dataset.train_cameras), opt_params)
    else
        nothing
    end

    sky = setup_sky_dome(rast, dataset, opt_params)
    sky_loss = setup_sky_supervision(sky, dataset, opt_params)
    normals = setup_normal_supervision(rast, opt_params)
    masks = setup_masking(dataset, opt_params)

    Trainer(
        rast, gs, dataset, ViewLoader(dataset), optimizers, cache,
        points_lr_scheduler, opt_params, strategy, densify, step, ids,
        depth_anchors, bilateral_grid, sky, sky_loss, normals, masks, LossLog())
end

# `gaussians` is excluded, as in `unsafe_free!`: the model is shared with
# whoever built the trainer and is accounted for there.
memory_usage(trainer::Trainer) =
    sum(memory_usage, values(trainer.optimizers); init=0) +
    Int(sizeof(trainer.cache)) +
    memory_usage(trainer.strategy) +
    memory_usage(trainer.rast) +
    memory_usage(trainer.bilateral_grid) +
    memory_usage(trainer.sky)

function KA.unsafe_free!(trainer::Trainer)
    foreach(free_optimizer!, values(trainer.optimizers))
    GPUArrays.unsafe_free!(trainer.cache)
    KA.unsafe_free!(trainer.strategy)
    KA.unsafe_free!(trainer.rast)
    isnothing(trainer.bilateral_grid) ||
        KA.unsafe_free!(trainer.bilateral_grid)
    isnothing(trainer.sky) || KA.unsafe_free!(trainer.sky)
    return
end

function setup_normal_supervision(rast::GaussianRasterizer, opt_params::OptimizationParams)
    opt_params.use_normal_loss || return false
    if rast.mode != :rgbdn
        @warn "Geometry regularization requires a `:rgbdn` rasterizer, disabling."
        return false
    end
    @info "Use geometry regularization (depth-normal consistency & flattening)."
    return true
end

function setup_depth_supervision(rast::GaussianRasterizer, dataset::ColmapDataset, opt_params::OptimizationParams)
    disabled = Maybe{DepthAnchor}[]
    if rast.mode != :rgbd && rast.mode != :rgbdn
        @warn "Depth supervision requires a `:rgbd` rasterizer, disabling."
        return disabled
    end
    if !dataset.has_depth_priors
        @warn "Depth supervision enabled, but no depth priors were found, disabling."
        return disabled
    end
    points = dataset.points
    if isempty(points)
        @warn "Depth supervision requires an init point cloud to fit anchors, disabling."
        return disabled
    end
    @assert dataset.depths_dir ≢ nothing

    # `collect_anchor_samples` indexes the prior at the camera's resolution, so
    # this one path wants it expanded on the host — it runs once & is cached.
    width, height = view_resolution(dataset)
    load_prior(i) = begin
        path = dataset.train_depth_paths[i]
        path ≡ nothing && return nothing
        return fit_resolution(load_depth_prior(path, width, height)[1], (width, height))
    end
    return load_or_fit_depth_anchors(
        dataset.depths_dir,
        points, dataset.train_cameras, load_prior;
        mode=opt_params.depth_loss_mode)
end

"""
Mean color of the top eighth of the train images: a rough sky color to start
the dome from, so it does not have to climb out of mid-grey while the scene is
also forming.

One streaming pass over the images, since the dataset no longer holds them —
the only place that still reads all of them at once, and only when the dome is
enabled.
"""
function sky_init_color(dataset::ColmapDataset)
    n_views = length(dataset)
    n_views == 0 && return SVector{3, Float32}(0.5f0, 0.5f0, 0.5f0)

    width, height = view_resolution(dataset)
    band_height = max(1, height ÷ 8)

    totals = zeros(Float64, 3, n_views)
    Threads.@threads for idx in 1:n_views
        image = load_image(dataset, idx, :train)
        totals[:, idx] .= vec(sum(Float64, @view(image[:, :, 1:band_height]); dims=(2, 3)))
    end
    mean_color = vec(sum(totals; dims=2)) ./ (255.0 * width * band_height * n_views)
    return SVector{3, Float32}(Float32.(mean_color)...)
end

function setup_sky_dome(
    rast::GaussianRasterizer, dataset::ColmapDataset, opt_params::OptimizationParams,
)
    opt_params.use_sky_dome || return nothing
    if rast.mode == :rgb
        @warn "Sky dome requires a `:rgbd` rasterizer for the alpha channel, disabling."
        return nothing
    end
    if isempty(dataset.train_cameras)
        @warn "Sky dome requires at least one train camera, disabling."
        return nothing
    end

    extent = dataset.camera_extent
    radius = sky_dome_radius(rast, opt_params, extent)
    requested = opt_params.sky_dome_radius_factor * extent
    radius < requested && @warn string(
        "Sky dome radius clamped to $(round(radius; digits=2)) from ",
        "$(round(requested; digits=2)) by the rasterizer far plane ",
        "($(rast.far_plane)); the dome will show some parallax.")

    center = sum(c -> c.camera_center, dataset.train_cameras) / length(dataset.train_cameras)
    # Orients the horizon cut for a `:hemisphere`; unused for a `:sphere`.
    up = estimate_up_vec(dataset.train_cameras)

    sky = SkyDome(
        get_backend(rast), dataset.train_cameras[1], opt_params;
        center, radius, up, color=sky_init_color(dataset))
    @info string(
        "Using a `$(opt_params.sky_dome_shape)` sky dome: $(length(sky)) ",
        "Gaussians at radius $(round(radius; digits=2)) ",
        "($(round(radius / extent; digits=1)) scene extents).")
    return sky
end

function setup_sky_supervision(sky::Maybe{SkyDome}, dataset::ColmapDataset, opt_params::OptimizationParams)
    opt_params.use_sky_loss || return false
    dataset.has_sky_masks || return false
    if sky ≡ nothing
        @warn "Sky masks found, but the sky dome is off: sky supervision disabled."
        return false
    end
    @info "Use sky masks to supervise the alpha map."
    return true
end

function setup_masking(dataset::ColmapDataset, opt_params::OptimizationParams)
    opt_params.use_masks || return false
    dataset.has_masks || return false
    @info "Use coverage masks: targets are blacked out beyond them."
    return true
end

"""
A loaded view's coverage mask on the device at the training resolution, or
`nothing` when masking is off or that view has none (see `masking.jl`).

Expanded per view by [`device_map`](@ref), like the sky masks & depth priors:
keeping every mask resident would cost a second copy of the dataset in device
memory, and expanding it on the host would cost the transfer of one.
"""
function view_mask(trainer::Trainer, mask::Maybe{Matrix{Float32}})
    (trainer.masks && mask ≢ nothing) || return nothing
    width, height = view_resolution(trainer.dataset)
    return device_map(get_backend(trainer.gaussians), mask, width, height)
end

# An `Adam` holds one moment pair per parameter array, so they are numbered.
function write_state!(tensors, meta, prefix::String, opt::NU.Adam)
    for (i, (μ, ν)) in enumerate(zip(opt.μ, opt.ν))
        tensors["$prefix.mu.$i"] = adapt(CPU(), μ)
        tensors["$prefix.nu.$i"] = adapt(CPU(), ν)
    end
    write_scalar!(meta, "$prefix.n_moments", length(opt.μ))
    write_scalar!(meta, "$prefix.current_step", opt.current_step)
    return
end

function read_state!(opt::NU.Adam, ckpt::Checkpoint, prefix::String)
    kab = get_backend(opt)
    n = read_scalar(ckpt, "$prefix.n_moments", Int)
    opt.μ = [adapt(kab, tensor(ckpt, "$prefix.mu.$i")) for i in 1:n]
    opt.ν = [adapt(kab, tensor(ckpt, "$prefix.nu.$i")) for i in 1:n]
    opt.current_step = read_scalar(ckpt, "$prefix.current_step", UInt32)
    return
end

const OPTIMIZER_NAMES = (
    :points, :features_dc, :features_rest, :opacities, :scales, :rotations)

function save_state(trainer::Trainer, filename::String)
    tensors = Dict{String, AbstractArray}()
    meta = Dict{String, String}()

    write_state!(tensors, meta, "gaussians", trainer.gaussians)
    for name in OPTIMIZER_NAMES
        write_state!(tensors, meta,
            "optimizers.$name", getproperty(trainer.optimizers, name))
    end

    if trainer.bilateral_grid ≢ nothing
        tensors["bilateral.grids"] = adapt(CPU(), trainer.bilateral_grid.grids)
        write_state!(tensors, meta,
            "bilateral.opt", trainer.bilateral_grid.optimizer)
    end

    trainer.sky ≡ nothing ||
        write_state!(tensors, meta, "sky", trainer.sky)

    write_state!(tensors, meta, "camera", trainer.dataset.train_cameras[1])
    write_scalar!(meta, "step", trainer.step)

    save_checkpoint(filename, tensors, meta)
    return
end

function load_state!(trainer::Trainer, filename::String)
    ckpt = load_checkpoint(filename)
    read_state!(trainer.gaussians, ckpt, "gaussians")
    for name in OPTIMIZER_NAMES
        read_state!(getproperty(trainer.optimizers, name), ckpt, "optimizers.$name")
    end

    # A checkpoint with grids restores them only if the trainer has them
    # enabled (they must match the same dataset).
    if trainer.bilateral_grid ≢ nothing && haskey(ckpt, "bilateral.grids")
        kab = get_backend(trainer.gaussians)
        copyto!(trainer.bilateral_grid.grids,
            adapt(kab, tensor(ckpt, "bilateral.grids")))
        read_state!(trainer.bilateral_grid.optimizer, ckpt, "bilateral.opt")
    end

    # Same as the grids: restored only when both sides have a dome, since its
    # geometry is derived from the dataset the checkpoint was trained on.
    if trainer.sky ≢ nothing && haskey(ckpt, "sky.gaussians.points")
        read_state!(trainer.sky, ckpt, "sky")
    end

    trainer.step = read_scalar(ckpt, "step", Int)
    return
end

# Convert image from UInt8 to Float32 & permute from (c, w, h) to (w, h, c, 1).
function device_target(trainer::Trainer, image::Array{UInt8, 3})
    image = device_image(get_backend(trainer.gaussians), image)
    image = permutedims(image, (2, 3, 1))
    return reshape(image, size(image)..., 1)
end

# Read view `idx` from disk & convert it, for callers outside the train loop
# (which goes through `take_view!` to overlap the read with the previous step).
get_image(trainer::Trainer, idx::Integer, set::Symbol) =
    device_target(trainer, load_image(trainer.dataset, Int(idx), set))

"""
Average SSIM / MSE / PSNR over the test views, each metric computed per view &
then averaged (the reference implementation's reduction).

`quantize` rounds the render to 8-bit before scoring, which is what published
numbers measure - see [`quantize8`](@ref). It costs a bit of accuracy on the
metric, so it is off for the in-training readout & on for benchmarks.

Masked views are scored against their masked target (see `masking.jl`), so the
easy black region counts toward the numbers & they are not comparable to
published ones.
"""
function validate(trainer::Trainer; quantize::Bool = false)
    gs = trainer.gaussians
    rast = trainer.rast
    dataset = trainer.dataset

    eval_ssim = 0f0
    eval_mse = 0f0
    eval_psnr = 0f0
    isempty(dataset.test_cameras) &&
        return (; eval_ssim, eval_mse, eval_psnr)

    for (idx, camera) in enumerate(dataset.test_cameras)
        view = load_view(dataset, idx, :test)
        target_image = device_target(trainer, view.image)
        mask = view_mask(trainer, view.mask)
        # `apply_mask`, not `composite_mask`: the pass below renders over the
        # rasterizer's default black, whatever the training background was.
        mask ≡ nothing || (target_image = apply_mask(target_image, mask))

        image_features = rast(
            gs.points, gs.opacities, gs.scales,
            gs.rotations, gs.features_dc, gs.features_rest;
            camera, sh_degree=gs.sh_degree)

        image = if rast.mode == :rgb
            image_features
        else
            image_features[1:3, :, :]
        end

        # The dome is part of the render: without it every sky pixel is scored
        # against black and the metrics report a regression that is not there.
        if trainer.sky ≢ nothing
            image = composite_sky(
                image, image_features[5, :, :], render_sky(trainer.sky, camera))
        end

        # From (c, w, h) to (w, h, c, 1) for SSIM.
        image_tmp = permutedims(image, (2, 3, 1))
        image_eval = reshape(image_tmp, size(image_tmp)..., 1)
        quantize && (image_eval = quantize8(image_eval))

        view_mse = mse(image_eval, target_image)
        eval_ssim += mean(fused_ssim(image_eval; ref=target_image))
        eval_mse += view_mse
        eval_psnr += psnr_from_mse(view_mse)
    end
    eval_ssim /= length(dataset.test_cameras)
    eval_mse /= length(dataset.test_cameras)
    eval_psnr /= length(dataset.test_cameras)
    return (; eval_ssim, eval_mse, eval_psnr)
end

function nonfinite_gradient_report(trainer::Trainer, θ, θ_names, ∇, camera::Camera, view_idx::Int)
    gs = trainer.gaussians
    n = length(gs)
    io = IOBuffer()
    println(io,
        "Non-finite gradients at step `$(trainer.step)` " *
        "(train view `$view_idx`: `$(trainer.dataset.train_image_filenames[view_idx])`):")

    bad_ids = Int[]
    for i in 1:length(θ)
        isempty(θ[i]) && continue
        ∇ᵢ = reshape(∇[i], :, n)
        mask = reshape(any(.!isfinite.(∇ᵢ); dims=1), :)
        n_bad = sum(mask)
        n_bad == 0 && continue
        println(io, "  ∇$(θ_names[i]): non-finite for $n_bad / $n gaussians")
        isempty(bad_ids) && (bad_ids = Array(findall(mask))[1:min(end, 5)])
    end

    # Forward state of the first few offending Gaussians.
    if !isempty(bad_ids)
        R_w2c = SMatrix{3, 3, Float32}(camera.w2c[1:3, 1:3])
        t_w2c = SVector{3, Float32}(camera.w2c[1:3, 4])
        ids_gpu = adapt(get_backend(gs), bad_ids)
        P = Array(gs.points[:, ids_gpu])
        O = Array(gs.opacities[:, ids_gpu])
        S = Array(gs.scales[:, ids_gpu])
        Q = Array(gs.rotations[:, ids_gpu])
        radii = Array(trainer.rast.gstate.radii[ids_gpu])
        for (k, gid) in enumerate(bad_ids)
            p = SVector{3, Float32}(P[1, k], P[2, k], P[3, k])
            z = (R_w2c * p + t_w2c)[3]
            println(io,
                "  gaussian `$gid`: z_cam=`$z`, radii=`$(radii[k])`, " *
                "opacity_raw=`$(O[1, k])`, scales_raw=`$(S[:, k])`, " *
                "‖q‖=`$(norm(@view(Q[:, k])))`")
        end
    end
    return String(take!(io))
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

    n_views = length(trainer.dataset)
    (trainer.step - 1) % n_views == 0 && shuffle!(trainer.ids)

    position = (trainer.step - 1) % n_views + 1
    idx = trainer.ids[position]
    view = take_view!(trainer.loader, idx, :train)
    # Queue the views the next few steps will want, before any GPU work is issued,
    # so their reads overlap this step.
    prefetch_start = position + 1
    prefetch_end = min(position + trainer.loader.lookahead, n_views)
    prefetch!(trainer.loader, trainer.ids[prefetch_start:prefetch_end])

    camera = trainer.dataset.train_cameras[idx]
    sky = trainer.sky
    # The dome *is* the background when it is on, so the scene pass renders
    # over zeros & compositing supplies what is behind it.
    background = (params.random_background && sky ≡ nothing) ?
        rand(SVector{3, Float32}) :
        zeros(SVector{3, Float32})

    θ = (
        gs.points, gs.features_dc, gs.features_rest,
        gs.opacities, gs.scales, gs.rotations)

    anchor = isempty(trainer.depth_anchors) ? nothing : trainer.depth_anchors[idx]
    bgrid = trainer.bilateral_grid

    kab = get_backend(rast)
    GPUArrays.@cached trainer.cache begin
        # Coverage mask: the target is blacked out beyond it.
        mask = view_mask(trainer, view.mask)
        hard_mask = mask ≡ nothing ? nothing : mask_hard(mask)
        # Weight of the alpha term that empties the rest of the frame.
        # Black target alone would accept an opaque black gaussian instead.
        empty_weight =
            (mask ≡ nothing || trainer.step < params.mask_opacity_from_iter) ?
            nothing : 1f0 .- mask

        # Composite target image over the same background the pass renders through.
        # Required for `random_background` to work.
        target_image = device_target(trainer, view.image)
        mask ≡ nothing || (target_image = composite_mask(target_image, mask, background))

        # Depth supervision target for this view.
        depth_data = if anchor ≡ nothing
            nothing
        else
            prior = device_map(kab, view.depth, view_resolution(trainer.dataset)...)
            target, half_band, valid, far_extrap = depth_target(anchor, prior, view.depth_qstep)
            # If masks are present in dataset, use depth supervision only inside masks.
            hard_mask ≡ nothing || (valid = valid .& hard_mask)
            # Depth dominates early geometry formation, photometric loss wins fine detail late.
            decay = params.depth_loss_final_scale^clamp(
                Float32(trainer.step / params.depth_loss_steps), 0f0, 1f0)
            weight = params.depth_loss_weight * decay
            (; target, half_band, valid, far_extrap, weight)
        end

        # Geometry regularization starts once the geometry is roughly in place.
        normal_data = if !trainer.normals || trainer.step < params.normal_from_iter
            nothing
        else
            (; rays=pixel_rays(kab, camera), valid=hard_mask)
        end

        # Sky mask supervision, once the geometry is roughly in place.
        sky_weight = if !trainer.sky_loss || trainer.step < params.sky_loss_from_iter
            nothing
        else
            sky_mask = view.sky_mask
            if sky_mask ≡ nothing
                nothing
            else
                w = device_map(kab, sky_mask, view_resolution(trainer.dataset)...)
                # Take into account `mask` if present.
                mask ≡ nothing ? w : w .* mask
            end
        end

        loss, ∇ = Zygote.withgradient(
            θ...,
            bgrid ≡ nothing ? nothing : bgrid.grids,
            sky ≡ nothing ? nothing : sky.gaussians.features_dc,
        ) do means_3d, features_dc, features_rest, opacities, scales, rotations, bgrids, sky_features
            image_features = rast(
                means_3d, opacities, scales, rotations, features_dc, features_rest;
                camera, sh_degree=gs.sh_degree, background)

            # NOTE:
            # Unconditional slice (a no-op for `:rgb`), a branch whose
            # else-arm aliases `image_features` unsliced makes Zygote
            # mis-route the gradient of the alias past the `getindex`
            # pullback once the depth term adds a second use, crashing
            # gradient accumulation with a shape mismatch.
            image = image_features[1:3, :, :]
            # NOTE: Same as above.
            depth_img, alpha_img = if (
                depth_data ≡ nothing &&
                normal_data ≡ nothing &&
                sky_features ≡ nothing &&
                empty_weight ≡ nothing
            )
                nothing, nothing
            else
                image_features[4, :, :], image_features[5, :, :]
            end

            # Composite Sky Dome behind the scene.
            if sky_features ≢ nothing
                image = composite_sky(image, alpha_img, render_sky(sky, camera, sky_features))
            end
            # Per-view appearance correction before the photometric loss.
            if bgrids ≢ nothing
                image = bilateral_slice(image, bgrids[:, :, :, :, idx])
            end

            # From (c, w, h) to (w, h, c, 1) for SSIM.
            image_tmp = permutedims(image, (2, 3, 1))
            image_eval = reshape(image_tmp, size(image_tmp)..., 1)

            # Compute losses.
            l1 = mean(abs.(image_eval .- target_image))
            s = 1f0 - mean(fused_ssim(image_eval; ref=target_image))

            l1_term = (1f0 - params.λ_dssim) * l1
            ssim_term = params.λ_dssim * s
            reg_term = regularization_loss(trainer.strategy, opacities, scales)
            tv_term = 0f0
            depth_term = 0f0
            sky_term = 0f0
            mask_term = 0f0
            flatten_term = 0f0
            normal_term = 0f0

            total = l1_term + ssim_term + reg_term

            if bgrids ≢ nothing
                tv_term = params.tv_loss_weight * tv_loss(bgrids)
                total += tv_term
            end

            if depth_data ≢ nothing
                depth_term =
                    depth_data.weight *
                    ssi_depth_loss(
                        depth_img, alpha_img;
                        depth_data.target, depth_data.half_band,
                        depth_data.valid, depth_data.far_extrap,
                        depth_floor=anchor.floor,
                        λ_grad=params.depth_loss_gradient_weight)
                total += depth_term
            end

            if sky_weight ≢ nothing
                sky_term = params.sky_loss_weight * zero_alpha_loss(alpha_img, sky_weight)
                total += sky_term
            end
            if empty_weight ≢ nothing
                mask_term = params.mask_opacity_weight * zero_alpha_loss(alpha_img, empty_weight)
                total += mask_term
            end

            if normal_data ≢ nothing
                flatten_term = params.normal_flatten_weight * flatten_loss(scales)
                normal_term =
                    params.normal_consistency_weight *
                    depth_normal_consistency_loss(
                        depth_img, alpha_img, image_features[6:8, :, :];
                        normal_data.rays, normal_data.valid)
                total += flatten_term + normal_term
            end

            # Save loss terms for logging.
            ignore_derivatives() do
                b = trainer.losses.current
                b.total = total
                b.l1 = l1_term
                b.ssim = ssim_term
                b.reg = reg_term
                b.tv = tv_term
                b.depth = depth_term
                b.sky = sky_term
                b.mask = mask_term
                b.flatten = flatten_term
                b.normal = normal_term
                nothing
            end
            total
        end

        # Guard before applying gradients: a non-finite loss means non-finite
        # gradients, which would irreversibly corrupt the parameters.
        isfinite(loss) || error(
            "Loss is not finite (`$loss`) at step `$(trainer.step)` " *
            "(train view `$idx`: `$(trainer.dataset.train_image_filenames[idx])`).")

        # Outside the closure: no reason to put this near the tape.
        update_ema!(trainer.losses)
        record!(trainer.losses.history, trainer.losses, trainer.step)

        gsp_debug = GSP_DEBUG[]

        # Apply gradients.
        θ_names = (:points, :features_dc, :features_rest, :opacities, :scales, :rotations)
        radii = rast.gstate.radii
        for i in 1:length(θ)
            θᵢ = θ[i]
            isempty(θᵢ) && continue
            ∇ᵢ = ∇[i]
            if gsp_debug
                # NaN/Inf gradients can arise even from a finite loss (e.g. `0·Inf`
                # in a pullback): catch them before the update corrupts parameters.
                # `sum` propagates non-finite values in a single cheap reduction.
                isfinite(sum(∇ᵢ)) || error(nonfinite_gradient_report(trainer, θ, θ_names, ∇, camera, idx))
            end
            if trainer.opt_params.use_sparse_adam
                sparse_step!(trainer.optimizers[i], θᵢ, ∇ᵢ, radii; dispose=false)
            else
                NU.step!(trainer.optimizers[i], θᵢ, ∇ᵢ; dispose=false)
            end
        end

        # Trailing `withgradient` arguments, in the order they are passed.
        ∇grids, ∇sky = ∇[end - 1], ∇[end]

        if bgrid ≢ nothing
            if gsp_debug
                isfinite(sum(∇grids)) || error(
                    "Non-finite bilateral grid gradient at step `$(trainer.step)` " *
                    "(train view `$idx`: `$(trainer.dataset.train_image_filenames[idx])`).")
            end
            NU.step!(bgrid.optimizer, bgrid.grids, ∇grids; dispose=false)
        end

        # Only the dome's colors are trainable; its geometry stays frozen so it
        # cannot drift into the scene & become a floater itself.
        if sky ≢ nothing
            if gsp_debug
                isfinite(sum(∇sky)) || error(
                    "Non-finite sky dome gradient at step `$(trainer.step)` " *
                    "(train view `$idx`: `$(trainer.dataset.train_image_filenames[idx])`).")
            end
            NU.step!(sky.optimizer, sky.gaussians.features_dc, ∇sky; dispose=false)
        end
    end

    if trainer.densify
        post_train_step!(
            trainer.strategy, gs, trainer.optimizers, rast, camera, trainer.cache;
            trainer.step, extent=trainer.dataset.camera_extent)
    end
    return loss
end

function update_lr!(trainer::Trainer)
    trainer.optimizers.points.lr = trainer.points_lr_scheduler(trainer.step)
    bgrid = trainer.bilateral_grid
    bgrid ≡ nothing || (bgrid.optimizer.lr = bgrid.scheduler(trainer.step))
    return
end
