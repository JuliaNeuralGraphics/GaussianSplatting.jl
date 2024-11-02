function project(
    means_3d::AbstractMatrix{Float32},
    scales::AbstractMatrix{Float32},
    rotations::AbstractMatrix{Float32};
    rast::GaussianRasterizer, camera::Camera,
    near_plane::Float32, far_plane::Float32,
    radius_clip::Float32, blur_ϵ::Float32,
)
    (; width, height) = resolution(camera)
    @assert width % 16 == 0 && height % 16 == 0

    K = camera.intrinsics
    R_w2c = SMatrix{3, 3, Float32}(camera.w2c[1:3, 1:3])
    t_w2c = SVector{3, Float32}(camera.w2c[1:3, 4])

    kab = get_backend(rast)
    n = size(means_3d, 2)

    means_2d = KA.zeros(kab, Float32, (2, n))
    conics = KA.zeros(kab, Float32, (3, n))
    compensations = KA.zeros(kab, Float32, (1, rast.antialias ? n : 0))

    if length(rast.gstate) < n
        KA.unsafe_free!(rast.gstate)
        rast.gstate = GeometryState(kab, n)
    end

    project!(kab, Int(BLOCK_SIZE))(
        # Output.
        rast.gstate.depths, # TODO support diff depth
        rast.gstate.radii,
        _as_T(SVector{2, Float32}, means_2d),
        _as_T(SVector{3, Float32}, conics),
        rast.antialias ? compensations : nothing,
        # Input Gaussians.
        _as_T(SVector{3, Float32}, means_3d),
        _as_T(SVector{3, Float32}, scales),
        _as_T(SVector{4, Float32}, rotations),
        # Input camera properties.
        R_w2c, t_w2c, K.focal, Int32.(K.resolution), K.principal,
        # Config.
        near_plane, far_plane, radius_clip, blur_ϵ; ndrange=n)
    return means_2d, conics, compensations
end

function ∇project(
    vmeans_2d::AbstractMatrix{Float32},
    vconics::AbstractMatrix{Float32},
    vcompensations, # TODO Union{ZeroTangent, AbstractVector{Float32}},
    # FWD input.
    means_3d::AbstractMatrix{Float32},
    scales::AbstractMatrix{Float32},
    rotations::AbstractMatrix{Float32},
    compensations::AbstractMatrix{Float32},
    conics::AbstractMatrix{Float32};
    rast::GaussianRasterizer, camera::Camera,
    near_plane::Float32, far_plane::Float32,
    radius_clip::Float32, blur_ϵ::Float32,
)
    K = camera.intrinsics
    R_w2c = SMatrix{3, 3, Float32}(camera.w2c[1:3, 1:3])
    t_w2c = SVector{3, Float32}(camera.w2c[1:3, 4])

    kab = get_backend(rast)
    n = size(means_3d, 2)

    vmeans = KA.zeros(kab, Float32, (3, n))
    vscales = KA.zeros(kab, Float32, (3, n))
    vrot = KA.zeros(kab, Float32, (4, n))

    # TODO check that vcompensations is not ZeroTangent if antialias
    ∇project!(kab, Int(BLOCK_SIZE))(
        # Output.
        _as_T(SVector{3, Float32}, vmeans),
        _as_T(SVector{3, Float32}, vscales),
        _as_T(SVector{4, Float32}, vrot),

        # Input grad outputs.
        _as_T(SVector{2, Float32}, vmeans_2d),
        _as_T(SVector{3, Float32}, vconics),
        vcompensations,

        _as_T(SVector{3, Float32}, conics),
        rast.gstate.radii,
        # Input Gaussians.
        _as_T(SVector{3, Float32}, means_3d),
        _as_T(SVector{3, Float32}, scales),
        _as_T(SVector{4, Float32}, rotations),
        rast.antialias ? compensations : nothing,
        # Input camera properties.
        R_w2c, t_w2c, K.focal, Int32.(K.resolution), K.principal,
        blur_ϵ; ndrange=n)

    # Accumulate for densificaton.
    @view(rast.gstate.∇means_2d[1:n]) .+= _as_T(SVector{2, Float32}, vmeans_2d)
    return vmeans, vscales, vrot
end

function ChainRulesCore.rrule(::typeof(project),
    means_3d::AbstractMatrix{Float32},
    scales::AbstractMatrix{Float32},
    rotations::AbstractMatrix{Float32};

    rast::GaussianRasterizer, camera::Camera,
    near_plane::Float32, far_plane::Float32,
    radius_clip::Float32, blur_ϵ::Float32,
)
    means_2d, conics, compensations = project(
        means_3d, scales, rotations;
        rast, camera, near_plane, far_plane,
        radius_clip, blur_ϵ)

    function _project_pullback(Ω)
        vmeans_2d, vconics, vcompensations = Ω
        ∇ = ∇project(
            vmeans_2d, vconics, vcompensations,
            means_3d, scales, rotations, compensations, conics;
            rast, camera, near_plane, far_plane,
            radius_clip, blur_ϵ)
        return (NoTangent(), ∇...)
    end
    return (means_2d, conics, compensations), _project_pullback
end

function spherical_harmonics(
    means_3d::AbstractMatrix{Float32},
    shs::AbstractArray{Float32, 3};
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
)
    kab = get_backend(rast)
    n = size(means_3d, 2)

    colors = KA.zeros(kab, Float32, (3, n))
    spherical_harmonics!(kab, Int(BLOCK_SIZE))(
        # Output.
        _as_T(SVector{3, Float32}, colors),
        rast.gstate.clamped,
        # Input.
        rast.gstate.radii,
        _as_T(SVector{3, Float32}, means_3d),
        camera.camera_center,
        reinterpret(SVector{3, Float32}, reshape(shs, :, n)),
        Val(sh_degree); ndrange=n)
    return colors
end

function ∇spherical_harmonics(
    vcolors::AbstractMatrix{Float32},
    means_3d::AbstractMatrix{Float32},
    shs::AbstractArray{Float32, 3};
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
)
    kab = get_backend(rast)
    n = size(means_3d, 2)

    vmeans_3d = KA.zeros(kab, Float32, size(means_3d))
    vshs = KA.zeros(kab, Float32, size(shs))
    ∇spherical_harmonics!(kab, Int(BLOCK_SIZE))(
        # Output.
        reinterpret(SVector{3, Float32}, reshape(vshs, :, n)),
        _as_T(SVector{3, Float32}, vmeans_3d),
        # Input.
        _as_T(SVector{3, Float32}, means_3d),
        reinterpret(SVector{3, Float32}, reshape(shs, :, n)),
        rast.gstate.clamped,
        _as_T(SVector{3, Float32}, vcolors),
        camera.camera_center,
        Val(sh_degree); ndrange=n)
    return vmeans_3d, vshs
end

function ChainRulesCore.rrule(::typeof(spherical_harmonics),
    means_3d::AbstractMatrix{Float32}, shs::AbstractArray{Float32, 3};
    rast::GaussianRasterizer, camera::Camera, sh_degree::Int,
)
    colors = spherical_harmonics(means_3d, shs; rast, camera, sh_degree)
    function _spherical_harmonics_pullback(vcolors)
        ∇ = ∇spherical_harmonics(vcolors, means_3d, shs; rast, camera, sh_degree)
        return (NoTangent(), ∇...)
    end
    return colors, _spherical_harmonics_pullback
end

function render(
    means_2d::AbstractMatrix{Float32},
    conics::AbstractMatrix{Float32},
    opacities::AbstractMatrix{Float32},
    colors::AbstractMatrix{Float32};
    rast::GaussianRasterizer, camera::Camera, background::SVector{3, Float32},
)
    kab = get_backend(rast)
    n = size(means_2d, 2)

    (; width, height) = resolution(camera)
    @assert width % 16 == 0 && height % 16 == 0

    image = KA.zeros(kab, Float32, (3, width, height))

    count_tiles_per_gaussian!(kab, Int(BLOCK_SIZE))(
        # Output.
        rast.gstate.tiles_touched,
        # Input.
        _as_T(SVector{2, Float32}, means_2d),
        rast.gstate.radii, rast.grid, BLOCK; ndrange=n)

    cumsum!(
        @view(rast.gstate.points_offset[1:n]),
        @view(rast.gstate.tiles_touched[1:n]))
    # Get total number of tiles touched.
    n_rendered = Int(@allowscalar rast.gstate.points_offset[n])
    n_rendered == 0 && return image

    if length(rast.bstate) < n_rendered
        KA.unsafe_free!(rast.bstate)
        rast.bstate = BinningState(kab, n_rendered)
    end

    # For each instance to be rendered, produce [tile | depth] key
    # and corresponding duplicated Gaussian indices to be sorted.
    duplicate_with_keys!(kab, Int(BLOCK_SIZE))(
        # Output.
        rast.bstate.gaussian_keys_unsorted,
        rast.bstate.gaussian_values_unsorted,
        # Input.
        _as_T(SVector{2, Float32}, means_2d),
        rast.gstate.depths,
        rast.gstate.points_offset,
        rast.gstate.radii, rast.grid, BLOCK; ndrange=n)

    sortperm!(
        @view(rast.bstate.permutation[1:n_rendered]),
        @view(rast.bstate.gaussian_keys_unsorted[1:n_rendered]))
    _permute!(kab)(
        rast.bstate.gaussian_keys_sorted, rast.bstate.gaussian_keys_unsorted,
        rast.bstate.permutation; ndrange=n_rendered)
    _permute!(kab)(
        rast.bstate.gaussian_values_sorted, rast.bstate.gaussian_values_unsorted,
        rast.bstate.permutation; ndrange=n_rendered)

    # Identify start-end of per-tile workloads in sorted keys.
    fill!(rast.istate.ranges, 0u32)
    identify_tile_range!(kab, Int(BLOCK_SIZE))(
        rast.istate.ranges, rast.bstate.gaussian_keys_sorted;
        ndrange=n_rendered)

    render!(kab, (Int.(BLOCK)...,), (width, height))(
        # Outputs.
        image,
        rast.istate.n_contrib,
        rast.istate.accum_α,
        # Inputs.
        rast.bstate.gaussian_values_sorted,
        _as_T(SVector{2, Float32}, means_2d),
        opacities,
        _as_T(SVector{3, Float32}, conics),
        _as_T(SVector{3, Float32}, colors),
        rast.gstate.depths,

        rast.istate.ranges,
        SVector{2, Int32}(width, height),
        background,
        BLOCK, Val(BLOCK_SIZE), Val(3i32))
    return image
end

function ∇render(
    vpixels::AbstractArray{Float32, 3},

    means_2d::AbstractMatrix{Float32},
    conics::AbstractMatrix{Float32},
    opacities::AbstractMatrix{Float32},
    colors::AbstractMatrix{Float32};
    rast::GaussianRasterizer, camera::Camera, background::SVector{3, Float32},
)
    kab = get_backend(rast)
    n = size(means_2d, 2)

    (; width, height) = resolution(camera)
    @assert width % 16 == 0 && height % 16 == 0

    fill!(reinterpret(Float32, rast.gstate.∇means_2d), 0f0)

    vmeans_2d = KA.zeros(kab, Float32, size(means_2d))
    vconics = KA.zeros(kab, Float32, size(conics))
    vopacities = KA.zeros(kab, Float32, size(opacities))
    vcolors = KA.zeros(kab, Float32, size(colors))

    ∇render!(kab, (Int.(BLOCK)...,), (width, height))(
        # Outputs.
        vcolors,
        vopacities,
        vconics,
        vmeans_2d,

        # Inputs.
        reshape(reinterpret(SVector{3, Float32}, vpixels), size(vpixels)[2:3]),
        rast.istate.n_contrib,
        rast.istate.accum_α,

        rast.bstate.gaussian_values_sorted,
        _as_T(SVector{2, Float32}, means_2d),
        opacities,
        _as_T(SVector{3, Float32}, conics),
        _as_T(SVector{3, Float32}, colors),

        rast.istate.ranges,
        SVector{2, Int32}(width, height), background,
        rast.grid, BLOCK, Val(BLOCK_SIZE), Val(3i32))

    # Accumulate for densificaton.
    @view(rast.gstate.∇means_2d[1:n]) .+= _as_T(SVector{2, Float32}, vmeans_2d)
    return vmeans_2d, vconics, vopacities, vcolors
end

function ChainRulesCore.rrule(::typeof(render),
    means_2d::AbstractMatrix{Float32},
    conics::AbstractMatrix{Float32},
    opacities::AbstractMatrix{Float32},
    colors::AbstractMatrix{Float32};
    rast::GaussianRasterizer, camera::Camera, background::SVector{3, Float32},
)
    image = render(
        means_2d, conics, opacities, colors;
        rast, camera, background)
    function _render_pullback(vpixels)
        ∇ = ∇render(
            vpixels, means_2d, conics, opacities, colors;
            rast, camera, background)
        return (NoTangent, ∇...)
    end
    return image, _render_pullback
end
