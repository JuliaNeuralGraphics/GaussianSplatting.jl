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
    _as_T(T, x) = reinterpret(T, reshape(x, :))

    means_2d = KA.allocate(kab, Float32, (2, n))
    conics = KA.allocate(kab, Float32, (3, n))

    length(rast.gstate) == n || (rast.gstate = GeometryState(kab, n))

    project!(kab, Int(BLOCK_SIZE))(
        # Output.
        rast.gstate.depths, # TODO support diff depth
        rast.gstate.radii,
        _as_T(SVector{2, Float32}, means_2d),
        _as_T(SVector{3, Float32}, conics),
        # Input Gaussians.
        _as_T(SVector{3, Float32}, means_3d),
        _as_T(SVector{3, Float32}, scales),
        _as_T(SVector{4, Float32}, rotations),
        # Input camera properties.
        R_w2c, t_w2c, K.focal, Int32.(K.resolution), K.principal,
        # Config.
        near_plane, far_plane, radius_clip, blur_ϵ; ndrange=n)

    return means_2d, conics
end

function ∇project(
    vmeans_2d::AbstractMatrix{Float32},
    vconics::AbstractMatrix{Float32},
    # FWD input.
    means_3d::AbstractMatrix{Float32},
    scales::AbstractMatrix{Float32},
    rotations::AbstractMatrix{Float32};

    rast::GaussianRasterizer, camera::Camera,
    near_plane::Float32, far_plane::Float32,
    radius_clip::Float32, blur_ϵ::Float32,
)
    K = camera.intrinsics
    R_w2c = SMatrix{3, 3, Float32}(camera.w2c[1:3, 1:3])
    t_w2c = SVector{3, Float32}(camera.w2c[1:3, 4])

    kab = get_backend(rast)
    n = size(means_3d, 2)
    _as_T(T, x) = reinterpret(T, reshape(x, :))

    vmeans = KA.zeros(kab, Float32, (3, n))
    vscales = KA.zeros(kab, Float32, (3, n))
    vrot = KA.zeros(kab, Float32, (4, n))

    # TODO ????
    # Accumulate for densificaton.
    rast.gstate.∇means_2d .+= _as_T(SVector{2, Float32}, vmeans_2d)

    ∇project!(kab, Int(BLOCK_SIZE))(
        # Output.
        _as_T(SVector{3, Float32}, vmeans),
        _as_T(SVector{3, Float32}, vscales),
        _as_T(SVector{4, Float32}, vrot),

        # Input grad outputs.
        _as_T(SVector{2, Float32}, vmeans_2d),
        _as_T(SVector{3, Float32}, vconics),

        rast.gstate.conic_opacities,
        rast.gstate.radii,
        # Input Gaussians.
        _as_T(SVector{3, Float32}, means_3d),
        _as_T(SVector{3, Float32}, scales),
        _as_T(SVector{4, Float32}, rotations),
        # Input camera properties.
        R_w2c, t_w2c, K.focal, Int32.(K.resolution), K.principal; ndrange=n)

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
    means_2d, conics = project(
        means_3d, scales, rotations;
        rast, camera, near_plane, far_plane,
        radius_clip, blur_ϵ)

    function _project_pullback(Ω)
        vmeans_2d, vconics = Ω
        ∇ = ∇project(
            vmeans_2d, vconics, means_3d, scales, rotations;
            rast, camera, near_plane, far_plane, radius_clip, blur_ϵ)
        return (NoTangent(), ∇...)
    end
    return (means_2d, conics), _project_pullback
end
