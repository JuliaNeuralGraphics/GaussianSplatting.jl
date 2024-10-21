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
    blur_ϵ::Float32,
)
    i = @index(Global)

    mean = means[i]
    mean_cam = pos_world_to_cam(R_w2c, t_w2c, mean)
    if !(near_plane < mean_cam[3] < far_plane)
        radii[i] = 0i32
        return
    end

    # Project Gaussian onto image plane.
    Σ = quat_scale_to_cov(cov_rotations[i], cov_scales[i])
    Σ_cam = covar_world_to_cam(R_w2c, Σ)
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
    conics[i] = SVector{4, Float32}(
        Σ_2D_inv[1, 1], Σ_2D_inv[2, 1], Σ_2D_inv[2, 2], 0f0) # TODO use SVector{3, Float32}
end

@kernel cpu=false function count_tiles_per_gaussian!(
    # Output.
    tiles_touched::AbstractVector{Int32},
    # Input.
    means_2D::AbstractVector{SVector{2, Float32}},
    radii::AbstractVector{Int32},
    tile_grid::SVector{2, Int32},
    tile_size::SVector{2, Int32},
)
    i = @index(Global)
    radius = radii[i]
    if !(radius > 0f0)
        tiles_touched[i] = 0i32
        return
    end

    mean_2D = means_2D[i]
    rect_min, rect_max = get_rect(mean_2D, radius, tile_grid, tile_size)
    area = prod(rect_max .- rect_min)
    tiles_touched[i] = area
end

@kernel cpu=false function render_v2!(
    # Output.
    out_color::AbstractArray{Float32, 3},
    n_contrib::AbstractMatrix{UInt32},
    accum_α::AbstractMatrix{Float32},
    # Input.
    gaussian_values_sorted::AbstractVector{UInt32},
    means_2d::AbstractVector{SVector{2, Float32}},
    opacities::AbstractMatrix{Float32},
    conics::AbstractVector{SVector{4, Float32}},
    rgb_features::AbstractVector{SVector{3, Float32}},
    depths::AbstractVector{Float32},

    ranges::AbstractMatrix{UInt32},
    resolution::SVector{2, Int32},
    background::SVector{3, Float32},
    block::SVector{2, Int32},
    ::Val{block_size}, ::Val{channels},
) where {block_size, channels}
    gidx = @index(Group, NTuple) # ≡ group_index
    lidx = @index(Local, NTuple) # ≡ thread_index
    ridx = @index(Local)         # ≡ thread_rank

    horizontal_blocks = gpu_cld(resolution[1], block[1])

    # Get current tile and starting pixel range (0-based indices).
    pix_min = SVector{2, Int32}(
        (gidx[1] - 1i32) * block[1],
        (gidx[2] - 1i32) * block[2])
    # 0-based indices.
    pix = SVector{2, Int32}(
        pix_min[1] + lidx[1] - 1i32,
        pix_min[2] + lidx[2] - 1i32)

    # Check if this thread corresponds to a valid pixel or is outside.
    # If not inside, this thread will help with data fetching,
    # but will not participate in rasterization.
    inside = pix[1] < resolution[1] && pix[2] < resolution[2]
    done::Bool = ifelse(inside, false, true)

    # Load start/end range of IDs to process.
    range_idx = (gidx[2] - 1i32) * horizontal_blocks + gidx[1]
    range = (Int32(ranges[1, range_idx]), Int32(ranges[2, range_idx]))
    to_do::Int32 = range[2] - range[1]
    # If `to_do` > `block_size`, repeat rasterization several times
    # with workitems in the workgroup.
    rounds::Int32 = gpu_cld(to_do, block_size)

    # Allocate storage for batches of collectively fetched data.
    collected_conics = @localmem SVector{4, Float32} block_size # TODO replace with 3
    collected_xy = @localmem SVector{2, Float32} block_size
    collected_opacity = @localmem Float32 block_size
    collected_id = @localmem UInt32 block_size

    T = 1f0
    contributor = 0u32
    last_contributor = 0u32

    color = zeros(MVector{3, Float32})

    for round in 0i32:(rounds - 1i32)
        # Collectively fetch data from global to shared memory.
        progress = range[1] + block_size * round + ridx # 1-based.

        @synchronize()
        if progress ≤ range[2]
            # TODO vectorize load with SIMD
            gaussian_id = gaussian_values_sorted[progress]
            collected_id[ridx] = gaussian_id
            collected_xy[ridx] = means_2d[gaussian_id]
            collected_opacity[ridx] = opacities[gaussian_id]
            collected_conics[ridx] = conics[gaussian_id]
        end
        @synchronize()
        # If `done`, this thread only helps with data fetching.
        done && continue

        for j in 1i32:min(block_size, to_do)
            # Keep track over current position in range.
            contributor += 1u32

            xy = collected_xy[j]
            opacity = collected_opacity[j]
            conic = collected_conics[j]
            δ = xy .- pix
            σ = conic[2] * δ[1] * δ[2] +
                0.5f0 * (conic[1] * δ[1]^2 + conic[3] * δ[2]^2)
            σ < 0f0 && continue

            α = min(0.99f0, opacity * exp(-σ))
            α < (1f0 / 255f0) && continue

            T_tmp = T * (1f0 - α)
            if T_tmp < 1f-4
                done = true
                break
            end

            gaussian_id = collected_id[j]
            feature = rgb_features[gaussian_id]
            @unroll for c in 1i32:channels
                color[c] += feature[c] * α * T
            end

            T = T_tmp

            # Keep track of last range entry to update this pixel.
            last_contributor = contributor
        end
        to_do -= block_size
    end

    if inside
        px, py = pix .+ 1i32
        accum_α[px, py] = T
        n_contrib[px, py] = last_contributor
        @unroll for c in 1i32:channels
            out_color[c, px, py] = color[c] + T * background[c]
        end
    end
end

function quat_scale_to_cov(q::SVector{4, Float32}, scale::SVector{3, Float32})
    S = sdiagm(scale...)
    R = unnorm_quat2rot(q)
    M = R * S
    return M * M'
end

function ∇quat_scale_to_cov(
    q::SVector{4, Float32}, scale::SVector{3, Float32},
    R::SMatrix{3, 3, Float32, 9}, vΣ::SMatrix{3, 3, Float32, 9},
)
    S = sdiagm(scale...)
    M = R * S

    vM = (vΣ + vΣ') * M
    vR = vM * S

    vq = ∇unnorm_quat2rot(q, vR)
    vscale = SVector{3, Float32}(
        R[1, 1] * vM[1, 1] + R[2, 1] * vM[2, 1] + R[3, 1] * vM[3, 1],
        R[1, 2] * vM[1, 2] + R[2, 2] * vM[2, 2] + R[3, 2] * vM[3, 2],
        R[1, 3] * vM[1, 3] + R[2, 3] * vM[2, 3] + R[3, 3] * vM[3, 3],
    )
    return vq, vscale
end

function unnorm_quat2rot(q::SVector{4, Float32})
    q = normalize(q)
    w, x, y, z = q
    x², y², z² = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return SMatrix{3, 3, Float32, 9}(
        1f0 - 2f0 * (y² + z²), 2f0 * (xy + wz), 2f0 * (xz - wy),
        2f0 * (xy - wz), 1f0 - 2f0 * (x² + z²), 2f0 * (yz + wx),
        2f0 * (xz + wy), 2f0 * (yz - wx), 1f0 - 2f0 * (x² + y²))
end

function ∇unnorm_quat2rot(q::SVector{4, Float32}, vR::SMatrix{3, 3, Float32, 9})
    inv_norm = 1f0 / norm(q)
    q = q / inv_norm
    w, x, y, z = q

    vqn = SVector{4, Float32}(
        2f0 * (
            x * (vR[3, 2] - vR[2, 3]) +
            y * (vR[1, 3] - vR[3, 1]) +
            z * (vR[2, 1] - vR[2, 1])
        ),
        2f0 * (
            -2f0 * x * (vR[2, 2] + vR[3, 3]) +
            y * (vR[2, 1] + vR[2, 1]) +
            z * (vR[3, 1] + vR[1, 3]) +
            w * (vR[3, 2] - vR[2, 3])
        ),
        2f0 * (
            x * (vR[2, 1] + vR[1, 2]) -
            2f0 * y * (vR[1, 1] + vR[3, 3]) +
            z * (vR[3, 2] + vR[2, 3]) +
            w * (vR[1, 3] - vR[3, 1])
        ),
        2f0 * (
            x * (vR[3, 1] + vR[1, 3]) +
            y * (vR[3, 2] + vR[2, 3]) -
            2f0 * z * (vR[1, 1] + vR[2, 2]) +
            w * (vR[2, 1] - vR[1, 2])
        ),
    )
    return (vqn - (vqn ⋅ q) * q) * inv_norm
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

function ∇inverse(x::SMatrix{2, 2, Float32, 4}, vx::SMatrix{2, 2, Float32, 4})
    return -x * vx * x
end

function add_blur(Σ_2D::SMatrix{2, 2, Float32, 4}, ϵ::Float32)
    det_orig = Σ_2D[1, 1] * Σ_2D[2, 2] - Σ_2D[1, 2] * Σ_2D[2, 1]
    Σ_2D = SMatrix{2, 2, Float32, 4}(
        Σ_2D[1, 1] + ϵ, Σ_2D[2, 1],
        Σ_2D[1, 2],     Σ_2D[2, 2] + ϵ,
    )
    det_blur = Σ_2D[1, 1] * Σ_2D[2, 2] - Σ_2D[1, 2] * Σ_2D[2, 1]
    compensation = sqrt(max(0f0, det_orig / det_blur))
    return Σ_2D, det_blur, compensation
end

# TODO
function ∇add_blur()

end
