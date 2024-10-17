@kernel cpu=false function project!(
    # Output.
    depths::AbstractVector{Float32},
    radii::AbstractVector{Int32},
    means_2D::AbstractVector{SVector{2, Float32}},
    # TODO use SVector{3, Float32}
    conics::AbstractVector{SVector{4, Float32}},

    # Input Gaussians.
    means::AbstractVector{SVector{3, Float32}},
    cov_scales::AbstractVector{SVector{3, Float32}},
    cov_rotations::AbstractVector{SVector{4, Float32}},

    # Input camera properties.
    R_w2c::SMatrix{3, 3, Float32, 9},
    t_w2c::SVector{3, Float32},
    focal::SVector{2, Float32},
    resolution::SVector{2, Int32},
    principal::SVector{2, Float32},

    # Config.
    near_plane::Float32,
    far_plane::Float32,
    radius_clip::Float32,
)
    i = @index(Global)

    mean = means[i]
    mean_cam = pos_world_to_cam(R_w2c, t_w2c, mean)
    if !(near_plane < mean_cam[3] < far_plane)
        radii[i] = 0i32
        return
    end

    Σ = quat_scale_to_cov(cov_rotations[i], cov_scales[i])
    Σ_cam = covar_world_to_cam(R_w2c, Σ)
    Σ_2D, mean_2D = perspective_projection(
        mean_cam, Σ_cam, focal, resolution, principal)

    # TODO add blur & compensation?
    # det = 0f0
    # if !(det > 0f0)
    #     radii[i] = 0i32
    #     return
    # end

    det, Σ_2D_inv = inverse(Σ_2D)# TODO use original det, before blur
    # Take 3σ as the radius.
    λ = max_eigval_2D(Σ_2D, det)
    radius = gpu_ceil(Int32, 3f0 * sqrt(λ))
    if radius ≤ radius_clip
        radii[i] = 0i32
        return
    end

    radii[i] = radius
    means_2D[i] = mean_2D
    depths[i] = mean_cam[3]
    conics[i] = SVector{4, Float32}(
        Σ_2D_inv[1, 1], Σ_2D_inv[2, 1], Σ_2D_inv[2, 2], 0f0)
end

function quat_scale_to_cov(q::SVector{4, Float32}, scale::SVector{3, Float32})
    S = sdiagm(scale...)
    R = unnorm_quat2rot(q)
    M = R * S
    return M * M'
end

function unnorm_quat2rot(q::SVector{4, Float32})
    q = q * inv(norm(q))
    w, x, y, z = q
    x², y², z² = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return SMatrix{3, 3, Float32, 9}(
        1f0 - 2f0 * (y² + z²), 2f0 * (xy + wz), 2f0 * (xz - wy),
        2f0 * (xy - wz), 1f0 - 2f0 * (x² + z²), 2f0 * (yz + wx),
        2f0 * (xz + wy), 2f0 * (yz - wx), 1f0 - 2f0 * (x² + y²),
    )
end

# TODO
function ∇unnorm_quat2rot()

end

function inverse(x::SMatrix{2, 2, Float32, 4})
    det = x[1, 1] * x[2, 2] - x[1, 2] * x[2, 1]
    if det ≈ 0f0
        return det, zeros(SMatrix{2, 2, Float32, 4})
    end

    det_inv = 1f0 / det
    tmp = -x[1, 2] * det_inv
    x_inv = SMatrix{2, 2, Float32, 4}(
        x[2, 2] * det_inv, tmp,
        tmp, x[1, 1] * det_inv,
    )
    return det, x_inv
end

# TODO
function ∇inverse()

end
