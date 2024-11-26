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
    depths = KA.zeros(kab, Float32, n)

    if length(rast.gstate) < n
        KA.unsafe_free!(rast.gstate)

        do_record = record_memory(kab)
        do_record && record_memory!(kab, false; free=false)
        rast.gstate = GeometryState(kab, n; extended=rast.mode == :rgbd)
        do_record && record_memory!(kab, true)
    end

    project!(kab)(
        # Output.
        depths,
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
    return means_2d, conics, compensations, depths
end

function ∇project(
    vmeans_2d::AbstractMatrix{Float32},
    vconics::AbstractMatrix{Float32},
    vcompensations, # TODO Union{ZeroTangent, AbstractVector{Float32}},
    vdepths, # TODO Union{ZeroTangent, AbstractVector{Float32}},
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
    ∇project!(kab)(
        # Output.
        _as_T(SVector{3, Float32}, vmeans),
        _as_T(SVector{3, Float32}, vscales),
        _as_T(SVector{4, Float32}, vrot),

        # Input grad outputs.
        _as_T(SVector{2, Float32}, vmeans_2d),
        _as_T(SVector{3, Float32}, vconics),
        vcompensations,
        vdepths,

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
    means_2d, conics, compensations, depths = project(
        means_3d, scales, rotations;
        rast, camera, near_plane, far_plane,
        radius_clip, blur_ϵ)

    function _project_pullback(Ω)
        vmeans_2d, vconics, vcompensations, vdepths = Ω
        ∇ = ∇project(
            vmeans_2d, vconics, vcompensations, vdepths,
            means_3d, scales, rotations, compensations, conics;
            rast, camera, near_plane, far_plane,
            radius_clip, blur_ϵ)
        return (NoTangent(), ∇...)
    end
    return (means_2d, conics, compensations, depths), _project_pullback
end

@kernel cpu=false inbounds=true function project!(
    # Output.
    depths::AbstractVector{Float32},
    radii::AbstractVector{Int32},
    means_2D::AbstractVector{SVector{2, Float32}},
    conics::AbstractVector{SVector{3, Float32}},
    compensations::C,

    # Input Gaussians.
    means::AbstractVector{SVector{3, Float32}},
    cov_scales::AbstractVector{SVector{3, Float32}},
    cov_rotations::AbstractVector{SVector{4, Float32}},

    # Input camera properties.
    R_w2c::RM, t_w2c,
    focal::SVector{2, Float32},
    resolution::SVector{2, Int32},
    principal::SVector{2, Float32},

    # Config.
    near_plane::Float32,
    far_plane::Float32,
    radius_clip::Float32,
    blur_ϵ::Float32,
) where {C <: Maybe{AbstractMatrix{Float32}}, RM}
    i = @index(Global)

    R, t = if RM <: StaticArray
        R_w2c, t_w2c
    else
        smat3f0(R_w2c), svec3f0(t_w2c)
    end

    mean = means[i]
    mean_cam = pos_world_to_cam(R, t, mean)
    if !(near_plane < mean_cam[3] < far_plane)
        radii[i] = 0i32
        return
    end

    # Project Gaussian onto image plane.
    cov_rotation = vload(pointer(cov_rotations, i)) # SIMD load
    Σ = quat_scale_to_cov(cov_rotation, cov_scales[i])
    Σ_cam = covar_world_to_cam(R, Σ)
    Σ_2D, mean_2D = perspective_projection(
        mean_cam, Σ_cam, focal, resolution, principal)

    Σ_2D, det, compensation = add_blur(Σ_2D, blur_ϵ)
    if !(det > 0f0)
        radii[i] = 0i32
        return
    end

    _, Σ_2D_inv = inverse(Σ_2D)

    # Take 3σ as the radius.
    λ = max_eigval_2D(Σ_2D, det)
    radius = gpu_ceil(Int32, 3f0 * sqrt(λ))
    if radius ≤ radius_clip
        radii[i] = 0i32
        return
    end

    # Discard Gaussians outside of image plane.
    if (
        (mean_2D[1] + radius) ≤ 0 ||
        (mean_2D[1] - radius) ≥ resolution[1] ||
        (mean_2D[2] + radius) ≤ 0 ||
        (mean_2D[2] - radius) ≥ resolution[2]
    )
        radii[i] = 0i32
        return
    end

    radii[i] = radius
    means_2D[i] = mean_2D
    depths[i] = mean_cam[3]
    conics[i] = SVector{3, Float32}(Σ_2D_inv[1, 1], Σ_2D_inv[2, 1], Σ_2D_inv[2, 2])
    if C <: AbstractMatrix{Float32}
        compensations[i] = compensation
    end
end

@kernel cpu=false inbounds=true function ∇project!(
    # Output.
    vmeans::AbstractVector{SVector{3, Float32}},
    vcov_scales::AbstractVector{SVector{3, Float32}},
    vcov_rotations::AbstractVector{SVector{4, Float32}},
    vR_out::RG, vt_out,

    # Input grad outputs.
    vmeans_2d::AbstractVector{SVector{2, Float32}},
    vconics::AbstractArray{SVector{3, Float32}},
    vcompensations::VC, #::AbstractVector{Float32},
    vdepths::VD,

    conics::AbstractVector{SVector{3, Float32}},
    radii::AbstractVector{Int32},

    # Input Gaussians.
    means::AbstractVector{SVector{3, Float32}},
    cov_scales::AbstractVector{SVector{3, Float32}},
    cov_rotations::AbstractVector{SVector{4, Float32}},
    compensations::C,

    # Input camera properties.
    R_w2c::RM, t_w2c,
    focal::SVector{2, Float32},
    resolution::SVector{2, Int32},
    principal::SVector{2, Float32},
    ϵ::Float32,
) where {C <: Maybe{AbstractMatrix{Float32}}, VC, VD, RM, RG}
    i = @index(Global)

    conic = conics[i]
    Σ_2D_inv = SMatrix{2, 2, Float32, 4}(
        conic[1], conic[2],
        conic[2], conic[3])

    vconic = vconics[i]
    vΣ_2D_inv = SMatrix{2, 2, Float32, 4}(
        vconic[1], vconic[2],
        vconic[2], vconic[3])

    vΣ_2D = ∇inverse(Σ_2D_inv, vΣ_2D_inv)

    if C <: AbstractMatrix{Float32} && VC <: AbstractMatrix{Float32}
        compensation = compensations[i]
        vcompensation = vcompensations[i]
        vΣ_2D = vΣ_2D + ∇add_blur(compensation, vcompensation, Σ_2D_inv, ϵ)
    end

    R, t = if RM <: StaticArray
        R_w2c, t_w2c
    else
        smat3f0(R_w2c), svec3f0(t_w2c)
    end

    mean = means[i]
    mean_cam = pos_world_to_cam(R, t, mean)

    cov_rotation = vload(pointer(cov_rotations, i))
    cov_scale = cov_scales[i]
    Σ = quat_scale_to_cov(cov_rotation, cov_scale)
    Σ_cam = covar_world_to_cam(R, Σ)

    vmean_2d = vmeans_2d[i]
    vΣ_cam, vmean_cam = ∇perspective_projection(
        mean_cam, Σ_cam,
        focal, resolution, principal,
        vΣ_2D, vmean_2d,
    )

    if VD <: AbstractVector{Float32}
        vdepth = vdepths[i]
        vmean_cam = SVector{3, Float32}(
            vmean_cam[1], vmean_cam[2], vmean_cam[3] + vdepth)
    end

    vR, vt, vmean = ∇pos_world_to_cam(R, t, mean, vmean_cam)
    vR, vΣ = ∇covar_world_to_cam(R, Σ, vΣ_cam, vR)
    vq, vscale = ∇quat_scale_to_cov(
        cov_rotation, cov_scale, unnorm_quat2rot(cov_rotation), vΣ)

    vmeans[i] = vmean
    vcov_scales[i] = vscale
    vstore!(pointer(vcov_rotations, i), vq) # SIMD store

    if RG != Nothing
        @unroll for rr in 1:3
            @unroll for rc in 1:3
                v = vR[rr, rc]
                if abs(v) > 1f-7 # For numerical stability.
                    @atomic vR_out[rr, rc] += v
                end
            end
            v = vt[rr]
            if abs(v) > 1f-7 # For numerical stability.
                @atomic vt_out[rr] += v
            end
        end
    end
end

@inbounds function perspective_projection(
    mean::SVector{3, Float32},
    Σ::SMatrix{3, 3, Float32, 9},
    focal::SVector{2, Float32},
    resolution::SVector{2, Int32},
    principal::SVector{2, Float32},
)::Tuple{SMatrix{2, 2, Float32, 4}, SVector{2, Float32}}
    tan_fov = 0.5f0 .* resolution ./ focal
    scaled_tan_fov = 0.3f0 .* tan_fov
    principal = principal .* resolution # convert from [0, 1] to [0, wh]

    rz = 1f0 / mean[3]
    rz² = rz * rz

    mean_xy = SVector{2, Float32}(mean[1], mean[2])
    mean_2D = rz .* focal .* mean_xy .+ principal

    lim_xy = (resolution .- principal) ./ focal .+ scaled_tan_fov
    lim_xy_neg = principal ./ focal .+ scaled_tan_fov
    txy = mean[3] .* min.(lim_xy, max.(-lim_xy_neg, mean_xy .* rz))

    J = SMatrix{2, 3, Float32, 6}(
        focal[1] * rz, 0f0,
        0f0, focal[2] * rz,
        -focal[1] * txy[1] * rz², -focal[2] * txy[2] * rz²)

    Σ_2D = J * Σ * J'
    return Σ_2D, mean_2D
end

@inbounds function ∇perspective_projection(
    mean::SVector{3, Float32},
    Σ::SMatrix{3, 3, Float32, 9},
    focal::SVector{2, Float32},
    resolution::SVector{2, Int32},
    principal::SVector{2, Float32},
    # Grad outputs.
    vΣ_2D::SMatrix{2, 2, Float32, 4},
    vmean_2D::SVector{2, Float32},
)::Tuple{SMatrix{3, 3, Float32, 9}, SVector{3, Float32}}
    tan_fov = 0.5f0 .* resolution ./ focal
    scaled_tan_fov = 0.3f0 .* tan_fov
    principal = principal .* resolution # convert from [0, 1] to [0, wh]

    rz = 1f0 / mean[3]
    rz² = rz * rz
    rz³ = rz² * rz

    lim_xy = (resolution .- principal) ./ focal .+ scaled_tan_fov
    lim_xy_neg = principal ./ focal .+ scaled_tan_fov
    mean_xy = SVector{2, Float32}(mean[1], mean[2])
    txy = mean[3] .* min.(lim_xy, max.(-lim_xy_neg, mean_xy .* rz))

    J = SMatrix{2, 3, Float32, 6}(
        focal[1] * rz, 0f0,
        0f0, focal[2] * rz,
        -focal[1] * txy[1] * rz², -focal[2] * txy[2] * rz²)

    vΣ::SMatrix{3, 3, Float32, 9} = J' * vΣ_2D * J
    vJ::SMatrix{2, 3, Float32, 6} =
        vΣ_2D  * J * Σ' +
        vΣ_2D' * J * Σ

    vmean = MVector{3, Float32}(
        focal[1] * rz * vmean_2D[1],
        focal[2] * rz * vmean_2D[2],
        -rz² * (
            focal[1] * mean[1] * vmean_2D[1] +
            focal[2] * mean[2] * vmean_2D[2]),
    )
    # FOV clipping.
    vmean[1] +=
        -lim_xy_neg[1] ≤ (mean[1] * rz) ≤ lim_xy[1] ?
        -focal[1] * rz² * vJ[1, 3] :
        -focal[1] * rz³ * vJ[1, 3] * txy[1]
    vmean[2] +=
        -lim_xy_neg[2] ≤ (mean[2] * rz) ≤ lim_xy[2] ?
        -focal[2] * rz² * vJ[2, 3] :
        -focal[2] * rz³ * vJ[2, 3] * txy[2]
    vmean[3] +=
        -focal[1] * rz² * vJ[1, 1] - focal[2] * rz² * vJ[2, 2] +
        2f0 * focal[1] * txy[1] * rz³ * vJ[1, 3] +
        2f0 * focal[2] * txy[2] * rz³ * vJ[2, 3]

    return vΣ, SVector{3, Float32}(vmean)
end

function pos_world_to_cam(
    R::SMatrix{3, 3, Float32, 9},
    t::SVector{3, Float32},
    point::SVector{3, Float32},
)
    return R * point + t
end

function ∇pos_world_to_cam(
    R::SMatrix{3, 3, Float32, 9},
    t::SVector{3, Float32},
    point::SVector{3, Float32},
    vpoint_cam::SVector{3, Float32},
)
    vR = vpoint_cam * point'
    vt = vpoint_cam
    vpoint = R' * vpoint_cam
    return vR, vt, vpoint
end

function covar_world_to_cam(
    R::SMatrix{3, 3, Float32, 9},
    Σ::SMatrix{3, 3, Float32, 9},
)
    return R * Σ * R'
end

function ∇covar_world_to_cam(
    R::SMatrix{3, 3, Float32, 9},
    Σ::SMatrix{3, 3, Float32, 9},
    vΣ_cam::SMatrix{3, 3, Float32, 9}, # grad out
    vR::SMatrix{3, 3, Float32, 9}, # grad in
)
    vR = vR +
        vΣ_cam  * R * Σ' +
        vΣ_cam' * R * Σ
    vΣ = R' * vΣ_cam * R
    return vR, vΣ
end
