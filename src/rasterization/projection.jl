function perspective_projection(
    point::SVector{3, Float32},
    Σ::SMatrix{3, 3, Float32, 9},
    focal::SVector{2, Float32},
    resolution::SVector{2, Int32},
    principal::SVector{2, Float32},
)::Tuple{SMatrix{2, 2, Float32, 4}, SVector{2, Float32}}
    tan_fov = 0.5f0 .* resolution ./ focal
    scaled_tan_fov = 0.3f0 .* tan_fov
    principal = principal .* resolution # convert from [0, 1] to [0, wh]

    lim_xy = (resolution .- principal) ./ focal .+ scaled_tan_fov
    lim_xy_neg = principal ./ focal .+ scaled_tan_fov

    rz = 1f0 / point[3]
    rz² = rz * rz

    point_xy = SVector{2, Float32}(point[1], point[2])
    point_2D = rz .* focal .* point_xy .+ principal

    txy = point[3] .* min.(lim_xy, max.(-lim_xy_neg, point_xy .* rz))
    J = SMatrix{2, 3, Float32, 6}(
        focal[1] * rz, 0f0,
        0f0, focal[2] * rz,
        -focal[1] * txy[1] * rz², -focal[2] * txy[2] * rz²)

    Σ_2D = J * Σ * J'
    return Σ_2D, point_2D
end

function ∇perspective_projection()

end
