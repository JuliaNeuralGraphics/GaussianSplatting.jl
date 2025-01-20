function densify_and_prune!(gs::GaussianModel, optimizers;
    extent::Float32, pruning_extent::Float32,
    grad_threshold::Float32, min_opacity::Float32,
    max_screen_size::Int32, dense_percent::Float32,
)
    ∇means_2d = gs.accum_∇means_2d ./ gs.denom
    mask = isnan.(∇means_2d)
    ∇means_2d[mask] .= 0f0
    KA.unsafe_free!(mask)

    densify_clone!(gs, optimizers; ∇means_2d, grad_threshold, extent, dense_percent)
    densify_split!(gs, optimizers; ∇means_2d, grad_threshold, extent, dense_percent)
    KA.unsafe_free!(∇means_2d)

    # Prune points that are too transparent, occupy too much space in image space
    # and have high scale in world space.
    valid_mask = reshape(NU.sigmoid.(gs.opacities) .> min_opacity, :)
    if max_screen_size > 0
        γ = 0.1f0 * pruning_extent
        valid_mask .&=
            (gs.max_radii .< max_screen_size) .&&
            reshape(maximum(exp.(gs.scales); dims=1) .< γ, :)
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
        reshape(maximum(exp.(gs.scales); dims=1) .< γ, :)

    new_points = gs.points[:, mask]
    new_features_dc = gs.features_dc[:, :, mask]
    new_features_rest = isempty(gs.features_rest) ?
        gs.features_rest : gs.features_rest[:, :, mask]
    new_scales = gs.scales[:, mask]
    new_rotations = gs.rotations[:, mask]
    new_opacities = gs.opacities[:, mask]

    new_ids = gs.ids ≡ nothing ? nothing : gs.ids[mask]
    KA.unsafe_free!(mask)

    densification_postfix!(gs, optimizers;
        new_points, new_features_dc, new_features_rest,
        new_scales, new_rotations, new_opacities, new_ids)

    KA.unsafe_free!(new_points)
    KA.unsafe_free!(new_features_dc)
    KA.unsafe_free!(new_features_rest)
    KA.unsafe_free!(new_scales)
    KA.unsafe_free!(new_rotations)
    KA.unsafe_free!(new_opacities)
    isnothing(new_ids) || KA.unsafe_free!(new_ids)
    return
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
        reshape(maximum(exp.(gs.scales); dims=1) .> γ, :)
    stds = repeat(exp.(gs.scales)[:, mask], 1, n_split)

    new_points = repeat(gs.points[:, mask], 1, n_split)
    new_features_rest = isempty(gs.features_rest) ?
        gs.features_rest : repeat(gs.features_rest[:, :, mask], 1, 1, n_split)
    new_features_dc = repeat(gs.features_dc[:, :, mask], 1, 1, n_split)
    new_scales = log.(stds ./ (0.8f0 * n_split))
    new_rotations = repeat(gs.rotations[:, mask], 1, n_split)
    new_opacities = repeat(gs.opacities[:, mask], 1, n_split)

    new_ids = gs.ids ≡ nothing ? nothing : repeat(gs.ids[mask], n_split)

    n_new_points = size(new_points, 2)
    if n_new_points > 0
        isotropic = size(gs.scales, 1) == 1
        _add_split_noise!(kab)(
            reinterpret(SVector{3, Float32}, new_points),
            reinterpret(SVector{4, Float32}, new_rotations),
            isotropic ? stds : reinterpret(SVector{3, Float32}, stds);
            ndrange=n_new_points)
    end
    KA.unsafe_free!(stds)

    densification_postfix!(gs, optimizers;
        new_points, new_features_dc, new_features_rest,
        new_scales, new_rotations, new_opacities, new_ids)

    KA.unsafe_free!(new_points)
    KA.unsafe_free!(new_features_dc)
    KA.unsafe_free!(new_features_rest)
    KA.unsafe_free!(new_scales)
    KA.unsafe_free!(new_rotations)
    KA.unsafe_free!(new_opacities)
    isnothing(new_ids) || KA.unsafe_free!(new_ids)

    # Prune gaussians that have small gradient or small size
    # ignoring newly inserted gaussians.
    valid_mask = vcat(.!mask, KA.ones(kab, Bool, n_new_points))
    KA.unsafe_free!(mask)
    prune_points!(gs, optimizers, valid_mask)
    return
end

@kernel cpu=false inbounds=true function _add_split_noise!(points, rots, stds)
    i = @index(Global)
    σ = stds[i]
    # ξ = SVector{3, Float32}(
    #     randn(Float32) * σ[1],
    #     randn(Float32) * σ[2],
    #     randn(Float32) * σ[3])
    ξ = σ .* SVector{3, Float32}(randn(Float32), randn(Float32), randn(Float32))

    q = rots[i]
    R = unnorm_quat2rot(q)
    p = points[i]
    points[i] = p .+ R * ξ
end

function prune_points!(gs::GaussianModel, optimizers, valid_mask)
    _prune_optimizer!(optimizers.points, valid_mask, gs.points)
    # TODO turn into macro:
    # @unsafe_replace gs.points = gs.points[:, valid_mask]
    new_points = gs.points[:, valid_mask]
    KA.unsafe_free!(gs.points)
    gs.points = new_points

    _prune_optimizer!(optimizers.features_dc, valid_mask, gs.features_dc)
    new_features_dc = gs.features_dc[:, :, valid_mask]
    KA.unsafe_free!(gs.features_dc)
    gs.features_dc = new_features_dc

    if !isempty(gs.features_rest)
        _prune_optimizer!(optimizers.features_rest, valid_mask, gs.features_rest)
        new_features_rest = gs.features_rest[:, :, valid_mask]
        KA.unsafe_free!(gs.features_rest)
        gs.features_rest = new_features_rest
    end

    _prune_optimizer!(optimizers.scales, valid_mask, gs.scales)
    new_scales = gs.scales[:, valid_mask]
    KA.unsafe_free!(gs.scales)
    gs.scales = new_scales

    _prune_optimizer!(optimizers.rotations, valid_mask, gs.rotations)
    new_rotations = gs.rotations[:, valid_mask]
    KA.unsafe_free!(gs.rotations)
    gs.rotations = new_rotations

    _prune_optimizer!(optimizers.opacities, valid_mask, gs.opacities)
    new_opacities = gs.opacities[:, valid_mask]
    KA.unsafe_free!(gs.opacities)
    gs.opacities = new_opacities

    new_max_radii = gs.max_radii[valid_mask]
    KA.unsafe_free!(gs.max_radii)
    gs.max_radii = new_max_radii

    new_accum_∇means_2d = gs.accum_∇means_2d[valid_mask]
    KA.unsafe_free!(gs.accum_∇means_2d)
    gs.accum_∇means_2d = new_accum_∇means_2d

    new_denom = gs.denom[valid_mask]
    KA.unsafe_free!(gs.denom)
    gs.denom = new_denom

    if gs.ids ≢ nothing
        new_ids = gs.ids[valid_mask]
        KA.unsafe_free!(gs.ids)
        gs.ids = new_ids
    end
    return
end

function densification_postfix!(
    gs::GaussianModel, optimizers;
    new_points, new_features_dc, new_features_rest,
    new_scales, new_rotations, new_opacities, new_ids,
)
    _append_optimizer!(optimizers.points, new_points)
    new_points = cat(gs.points, new_points; dims=ndims(new_points))
    KA.unsafe_free!(gs.points)
    gs.points = new_points

    _append_optimizer!(optimizers.features_dc, new_features_dc)
    new_features_dc = cat(gs.features_dc, new_features_dc; dims=ndims(new_features_dc))
    KA.unsafe_free!(gs.features_dc)
    gs.features_dc = new_features_dc

    if !isempty(gs.features_rest)
        _append_optimizer!(optimizers.features_rest, new_features_rest)
        new_features_rest = cat(gs.features_rest, new_features_rest; dims=ndims(new_features_rest))
        KA.unsafe_free!(gs.features_rest)
        gs.features_rest = new_features_rest
    end

    _append_optimizer!(optimizers.scales, new_scales)
    new_scales = cat(gs.scales, new_scales; dims=ndims(new_scales))
    KA.unsafe_free!(gs.scales)
    gs.scales = new_scales

    _append_optimizer!(optimizers.rotations, new_rotations)
    new_rotations = cat(gs.rotations, new_rotations; dims=ndims(new_rotations))
    KA.unsafe_free!(gs.rotations)
    gs.rotations = new_rotations

    _append_optimizer!(optimizers.opacities, new_opacities)
    new_opacities = cat(gs.opacities, new_opacities; dims=ndims(new_opacities))
    KA.unsafe_free!(gs.opacities)
    gs.opacities = new_opacities

    KA.unsafe_free!(gs.max_radii)
    KA.unsafe_free!(gs.accum_∇means_2d)
    KA.unsafe_free!(gs.denom)

    kab = get_backend(gs)
    n = size(gs.points, 2)
    gs.max_radii = KA.zeros(kab, Int32, n)
    gs.accum_∇means_2d = KA.zeros(kab, Float32, n)
    gs.denom = KA.zeros(kab, Float32, n)

    if gs.ids ≢ nothing
        new_ids = cat(gs.ids, new_ids; dims=1)
        KA.unsafe_free!(gs.ids)
        gs.ids = new_ids
    end
    return
end

function _append_optimizer!(opt::NU.Adam, extension)
    kab = get_backend(extension)
    dims = ndims(extension)

    μ = opt.μ[1]
    μ̂ = KA.zeros(kab, eltype(μ), size(extension))
    μ = cat(reshape(μ, size(extension)[1:end - 1]..., :), μ̂; dims)
    KA.unsafe_free!(opt.μ[1])
    opt.μ[1] = reshape(μ, :)

    ν = opt.ν[1]
    ν̂ = KA.zeros(kab, eltype(ν), size(extension))
    ν = cat(reshape(ν, size(extension)[1:end - 1]..., :), ν̂; dims)
    KA.unsafe_free!(opt.ν[1])
    opt.ν[1] = reshape(ν, :)
    return
end

function _prune_optimizer!(opt::NU.Adam, mask, x)
    d = ntuple(i -> Colon(), ndims(x) - 1)
    new_μ = reshape(reshape(opt.μ[1], size(x))[d..., mask], :)
    new_ν = reshape(reshape(opt.ν[1], size(x))[d..., mask], :)
    KA.unsafe_free!(opt.μ[1])
    KA.unsafe_free!(opt.ν[1])
    opt.μ[1] = new_μ
    opt.ν[1] = new_ν
    return
end
