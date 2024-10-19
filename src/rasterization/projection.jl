function perspective_projection(
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

function ∇perspective_projection(
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
    # grad out
    vpoint_cam::SVector{3, Float32},
    # grad in
    vR::SMatrix{3, 3, Float32, 9},
    vt::SVector{3, Float32},
    vpoint::SVector{3, Float32},
)
    vR = vR + vpoint_cam * point'
    vt = vt + vpoint_cam
    vpoint = vpoint + R' * vpoint_cam
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
    # grad out
    vΣ_cam::SMatrix{3, 3, Float32, 9},
    # grad in
    vR::SMatrix{3, 3, Float32, 9},
    vΣ::SMatrix{3, 3, Float32, 9},
)
    vR = vR +
        vΣ_cam  * R * Σ' +
        vΣ_cam' * R * Σ
    vΣ = vΣ + R' * vΣ_cam * R
    return vR, vΣ
end
