"""
Camera-space unit normal of a Gaussian: the rotation column of its smallest
axis, flipped to face the camera (PGSR-style, mirroring LichtFeld's
`primitive_normals`). The column of a rotation matrix is already unit-length,
so no normalization is needed.

The flip test uses `n_cam ⋅ mean_cam`, which equals LichtFeld's world-space
`axis ⋅ (mean - camera_center)` because `R_w2c` is orthonormal — and `mean_cam`
is already at hand in both projection kernels.

Returns `(n_cam, k, sign)`. Both `k` (the argmin) and `sign` are treated as
constants in the backward, so the only gradient is onto column `k` of `R_g`.
"""
@inline function gaussian_normal(
    R_w2c::SMatrix{3, 3, Float32, 9}, R_g::SMatrix{3, 3, Float32, 9},
    scale::SVector{3, Float32}, mean_cam::SVector{3, Float32},
)
    k = (scale[1] ≤ scale[2] && scale[1] ≤ scale[3]) ? 1i32 :
        (scale[2] ≤ scale[3]) ? 2i32 : 3i32
    axis =
        k == 1i32 ? SVector{3, Float32}(R_g[1, 1], R_g[2, 1], R_g[3, 1]) :
        k == 2i32 ? SVector{3, Float32}(R_g[1, 2], R_g[2, 2], R_g[3, 2]) :
                    SVector{3, Float32}(R_g[1, 3], R_g[2, 3], R_g[3, 3])
    n_cam = R_w2c * axis
    sign = (n_cam ⋅ mean_cam) > 0f0 ? -1f0 : 1f0
    return sign .* n_cam, k, sign
end

# Cotangent on `R_g` from the normal channel: `g` lands in column `k`.
@inline function ∇gaussian_normal_column(k::Int32, g::SVector{3, Float32})
    z = 0f0
    return k == 1i32 ?
        SMatrix{3, 3, Float32, 9}(g[1], g[2], g[3], z, z, z, z, z, z) :
        k == 2i32 ?
        SMatrix{3, 3, Float32, 9}(z, z, z, g[1], g[2], g[3], z, z, z) :
        SMatrix{3, 3, Float32, 9}(z, z, z, z, z, z, g[1], g[2], g[3])
end

@kernel cpu=false inbounds=true function project!(
    # Output.
    depths::AbstractVector{Float32},
    radii::AbstractVector{Int32},
    means_2D::AbstractVector{SVector{2, Float32}},
    conics::AbstractVector{SVector{3, Float32}},
    compensations::C,
    normals::N,

    # Input Gaussians.
    means::AbstractVector{SVector{3, Float32}},
    cov_scales::AbstractVector{SVector{3, Float32}},
    cov_rotations::AbstractVector{SVector{4, Float32}},
    # Activated opacities: the culling extent is opacity-aware (see `ellipse_radius_sq`).
    # So a faint Gaussian is culled sooner than an opaque one with the same covariance.
    opacities::AbstractMatrix{Float32},

    # Input camera properties.
    R_w2c::RM, t_w2c,
    focal::SVector{2, Float32},
    resolution::SVector{2, Int32},
    principal::SVector{2, Float32},

    # Config.
    near_plane::Float32,
    far_plane::Float32,
    radius_clip::Int32,
    blur_ϵ::Float32,
) where {
    C <: Maybe{AbstractMatrix{Float32}},
    N <: Maybe{AbstractVector{SVector{3, Float32}}},
    RM,
}
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
    cov_scale = cov_scales[i]
    R_g = unnorm_quat2rot(cov_rotation)
    Σ = quat_scale_to_cov(R_g, cov_scale)
    Σ_cam = covar_world_to_cam(R, Σ)
    Σ_2D, mean_2D = perspective_projection(
        mean_cam, Σ_cam, focal, resolution, principal)

    Σ_2D, det, compensation = add_blur(Σ_2D, blur_ϵ)
    if !(det > 0f0)
        radii[i] = 0i32
        return
    end

    radius_sq = ellipse_radius_sq(opacities[i])
    if !(radius_sq > 0f0)
        radii[i] = 0i32
        return
    end

    # Cull too small **on screen** Gaussians, use original 3σ radius.
    λ = max_eigval_2D(Σ_2D, det)
    if gpu_ceil(Int32, 3f0 * sqrt(λ)) ≤ radius_clip
        radii[i] = 0i32
        return
    end

    # Discard Gaussians outside of image plane.
    # The cull uses the anisotropic extent of the ellipse.
    _, Σ_2D_inv = inverse(Σ_2D)
    conic = SVector{3, Float32}(Σ_2D_inv[1, 1], Σ_2D_inv[2, 1], Σ_2D_inv[2, 2])
    extent = ellipse_extent(conic, radius_sq)
    if (
        (mean_2D[1] + extent[1]) ≤ 0f0 ||
        (mean_2D[1] - extent[1]) ≥ Float32(resolution[1]) ||
        (mean_2D[2] + extent[2]) ≤ 0f0 ||
        (mean_2D[2] - extent[2]) ≥ Float32(resolution[2])
    )
        radii[i] = 0i32
        return
    end

    radii[i] = gpu_ceil(Int32, sqrt(radius_sq * λ))
    means_2D[i] = mean_2D
    depths[i] = mean_cam[3]
    conics[i] = conic
    C <: AbstractMatrix{Float32} && (compensations[i] = compensation)
    if N <: AbstractVector{SVector{3, Float32}}
        normals[i], _, _ = gaussian_normal(R, R_g, cov_scale, mean_cam)
    end
end

@kernel cpu=false inbounds=true function ∇project!(
    # Output.
    vmeans::AbstractVector{SVector{3, Float32}},
    vcov_scales::AbstractVector{SVector{3, Float32}},
    vcov_rotations::AbstractVector{SVector{4, Float32}},
    vR_out::RG,
    vt_out,

    # Input grad outputs.
    vmeans_2d::AbstractVector{SVector{2, Float32}},
    vconics::AbstractArray{SVector{3, Float32}},
    vcompensations::VC,
    vdepths::VD,
    vnormals::VN,

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
) where {
    C <: Maybe{AbstractMatrix{Float32}},
    VC <: Maybe{AbstractVector{Float32}},
    VD <: Maybe{AbstractVector{Float32}},
    VN <: Maybe{AbstractVector{SVector{3, Float32}}},
    RM,
    RG <: Maybe{AbstractMatrix{Float32}},
}
    i = @index(Global)

    # Culled in the forward: cotangents are zero & `conics[i]`/`mean_cam`
    # are stale or degenerate (e.g. `z ≈ 0` → `1/z = Inf` → `0·Inf = NaN`
    # in `vmeans` for a Gaussian that was never rendered). Outputs are
    # zero-initialized, so returning leaves the correct zero gradient.
    radii[i] > 0i32 || return

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
    R_g = unnorm_quat2rot(cov_rotation)
    Σ = quat_scale_to_cov(R_g, cov_scale)
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

    # Rendered-normal channel: `n_cam = sign · R_w2c · R_g[:, k]`, so the
    # cotangent lands on that one column of `R_g` (`k` & `sign` are constants).
    # `vR_w2c` gets no contribution: pose optimization does not see the normals.
    vR_g = if VN <: AbstractVector{SVector{3, Float32}}
        _, k, sign = gaussian_normal(R, R_g, cov_scale, mean_cam)
        ∇gaussian_normal_column(k, sign .* (R' * vnormals[i]))
    else
        zeros(SMatrix{3, 3, Float32, 9})
    end
    vq, vscale = ∇quat_scale_to_cov(cov_rotation, cov_scale, R_g, vΣ, vR_g)

    vmeans[i] = vmean
    vcov_scales[i] = vscale
    vstore!(pointer(vcov_rotations, i), vq) # SIMD store

    # TODO reduce within warp/workgroup to decrease number of atomic ops.
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

    # NOTE: Accumulated in scalars rather than an `MVector`,
    # otherwise LLVM fails to emit correct code for CUDA.
    # LLVM builds the return value in the stack frame instead of in registers,
    # and it then merges two of the writes into one 8-byte store.
    # This function is not inlined into the kernel, so that store lands in the
    # caller's `sret` slot - which PTX only guarantees 4-byte alignment for
    # (`.local .align 4 __local_depot`), giving `ERROR_MISALIGNED_ADDRESS` on CUDA.
    vmean_x = focal[1] * rz * vmean_2D[1]
    vmean_y = focal[2] * rz * vmean_2D[2]
    vmean_z = -rz² * (
        focal[1] * mean[1] * vmean_2D[1] +
        focal[2] * mean[2] * vmean_2D[2])
    # FOV clipping: when clamped, `txy = z·lim` does not depend on `x` (`y`),
    # and the `J[·, 3]` contribution goes to `z` instead:
    # ∂J[1, 3]/∂z = f·txy·rz³ = 2f·txy·rz³ (below) - f·txy·rz³ (correction).
    if -lim_xy_neg[1] ≤ (mean[1] * rz) ≤ lim_xy[1]
        vmean_x += -focal[1] * rz² * vJ[1, 3]
    else
        vmean_z += -focal[1] * rz³ * vJ[1, 3] * txy[1]
    end
    if -lim_xy_neg[2] ≤ (mean[2] * rz) ≤ lim_xy[2]
        vmean_y += -focal[2] * rz² * vJ[2, 3]
    else
        vmean_z += -focal[2] * rz³ * vJ[2, 3] * txy[2]
    end
    vmean_z +=
        -focal[1] * rz² * vJ[1, 1] - focal[2] * rz² * vJ[2, 2] +
        2f0 * focal[1] * txy[1] * rz³ * vJ[1, 3] +
        2f0 * focal[2] * txy[2] * rz³ * vJ[2, 3]

    return vΣ, SVector{3, Float32}(vmean_x, vmean_y, vmean_z)
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
