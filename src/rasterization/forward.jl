# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
"""
Run preprocessing of gaussians: transforming, bounding, converting SH to RGB.

Note:
- P: number of gaussians (in PyTorch size of the 0th dim)
- M: number of spherical harmonics (in PyTorch size of the 1th dim) output size
- D: degree of spherical harmonics
"""
@kernel cpu=false inbounds=true function _preprocess!(
    # Outputs.
    cov3Ds, depths, radii, pixels, conic_opacities, tiles_touched, rgbs, clamped,
    # Inputs.
    means,
    scales,
    rotations,
    spherical_harmonics,
    sh_degree, opacities,
    projection, view, camera_position,
    resolution::SVector{2, Int32},
    grid, block,
    focal_xy::SVector{2, Float32},
    tan_fov_xy::SVector{2, Float32},
    principal::SVector{2, Float32},
    scale_modifier::Float32,
)
    i = @index(Global)
    radii[i] = 0i32
    tiles_touched[i] = 0i32

    point = means[i]
    point_h = to_homogeneous(point)
    visible, depth = in_frustum(point_h, view)
    if visible
        cov3D = computeCov3D(scales[i], rotations[i], scale_modifier)
        cov3Ds[i] = cov3D

        cov = computeCov2D(point_h, focal_xy, tan_fov_xy,
            resolution, principal, cov3D, view)
        det = cov[1] * cov[3] - cov[2]^2
        if det ≢ 0f0
            # Project point into camera space.
            projected_h = projection * point_h
            projected =
                SVector{3, Float32}(projected_h[1], projected_h[2], projected_h[3]) .*
                (1f0 / (projected_h[4] + eps(Float32)))

            # Compute inverse conic.
            det_inv = 1f0 / det
            conic = SVector{3, Float32}(
                cov[3] * det_inv, -cov[2] * det_inv, cov[1] * det_inv)

            # Compute extent in screen space (by finding eigenvalues of 2D covariance matrix).
            λ1, λ2 = eigvals_2D(cov, det)
            radius = gpu_ceil(Int32, 3f0 * sqrt(max(λ1, λ2)))
            # From `extent`, compute how many tiles does it cover in screen-space.
            pixel = SVector{2, Float32}(
                ndc2pix(projected[1], resolution[1]),
                ndc2pix(projected[2], resolution[2]))
            rmin, rmax = get_rect(pixel, radius, grid, block)
            # Quit if does not covert anything.
            area = (rmax[1] - rmin[1]) * (rmax[2] - rmin[2])
            if area > 0i32
                begin
                    depths[i] = depth
                    radii[i] = radius
                    pixels[i] = pixel
                    conic_opacities[i] = SVector{4, Float32}(
                        conic[1], conic[2], conic[3], opacities[i])
                    tiles_touched[i] = area

                    rgbs[i], clamped[i] = compute_colors_from_sh(
                        point, camera_position, @view(spherical_harmonics[:, i]), sh_degree)
                end
            end
        end
    end
end

@kernel cpu=false inbounds=true function render!(
    # Outputs.
    out_color::AbstractArray{Float32, 3},
    auxiliary::A,
    covisibility::C,
    n_contrib::AbstractMatrix{UInt32},
    accum_α::AbstractMatrix{Float32},
    # Inputs.
    gaussian_values_sorted::AbstractVector{UInt32},
    means_2d::AbstractVector{SVector{2, Float32}},
    conic_opacities::AbstractVector{SVector{4, Float32}},
    rgb_features::AbstractVector{SVector{3, Float32}},
    depths::AbstractVector{Float32},

    ranges::AbstractMatrix{UInt32},
    resolution::SVector{2, Int32},
    bg_color::SVector{3, Float32},
    block::SVector{2, Int32},
    ::Val{block_size}, ::Val{channels},
) where {A, C, block_size, channels}
    @uniform horizontal_blocks = gpu_cld(resolution[1], block[1])

    gidx = @index(Group, NTuple) # ≡ group_index
    lidx = @index(Local, NTuple) # ≡ thread_index
    ridx = @index(Local)         # ≡ thread_rank

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
    done = ifelse(inside, Cint(0), Cint(1))

    # Load start/end range of IDs to process.
    range_idx = (gidx[2] - 1i32) * horizontal_blocks + gidx[1]
    range = (Int32(ranges[1, range_idx]), Int32(ranges[2, range_idx]))
    to_do::Int32 = range[2] - range[1]
    # If `to_do` > `block_size`, repeat rasterization several times
    # with workitems in the workgroup.
    rounds::Int32 = gpu_cld(to_do, block_size)

    # Allocate storage for batches of collectively fetched data.
    collected_conic_opacity = @localmem SVector{4, Float32} block_size
    collected_xy = @localmem SVector{2, Float32} block_size
    collected_id = @localmem UInt32 block_size
    collected_depth = (A !== Nothing) ? (@localmem Float32 block_size) : nothing

    T = 1f0
    contributor = 0u32
    last_contributor = 0u32

    color = zeros(MVector{3, Float32})
    auxiliary_px = (A !== Nothing) ? zeros(MVector{3, Float32}) : nothing

    # Iterate over batches until done or range is complete.
    for round in 0i32:(rounds - 1i32)
        # Collectively fetch data from global to shared memory.
        progress = range[1] + block_size * round + ridx # 1-based.

        @synchronize()
        if progress ≤ range[2]
            gaussian_id = gaussian_values_sorted[progress]
            collected_id[ridx] = gaussian_id
            collected_xy[ridx] = means_2d[gaussian_id]
            collected_conic_opacity[ridx] = conic_opacities[gaussian_id]
            if A !== Nothing
                collected_depth[ridx] = depths[gaussian_id]
            end
        end
        @synchronize()

        # If `done`, this thread only helps with data fetching.
        done == Cint(1) && continue

        # Iterate over current batch.
        for j in 1i32:min(block_size, to_do)
            # Keep track over current position in range.
            contributor += 1u32

            # Resample using conic matrix:
            # ("Surface Splatting" by Zwicker et al., 2001).
            xy = collected_xy[j]
            con_o = collected_conic_opacity[j]
            power = gaussian_power(con_o, xy .- pix)
            power > 0f0 && continue

            # Eq. (2) from 3D Gaussian splatting paper.
            # Obtain alpha by multiplying with Gaussian opacity
            # and its exponential falloff from mean.
            # Avoid numerical instabilities (see paper appendix).
            α = min(0.99f0, con_o[4] * exp(power))
            α < (1f0 / 255f0) && continue

            T_tmp = T * (1f0 - α)
            if T_tmp < 1f-4
                done = Cint(1)
                break
            end

            gaussian_id = collected_id[j]
            # If needed, mark current Gaussian as visible.
            if C !== Nothing && T > 0.5f0
                covisibility[gaussian_id] = true
            end

            # Eq. (3) from 3D Gaussian splatting paper.
            feature = rgb_features[collected_id[j]]
            for c in 1i32:channels
                color[c] += feature[c] * α * T
            end
            if A !== Nothing
                auxiliary_px[1] += collected_depth[j] * α * T
                auxiliary_px[2] += α * T
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
        for c in 1i32:channels
            out_color[c, px, py] = color[c] + T * bg_color[c]
        end
        if A !== Nothing
            auxiliary[1, px, py] = auxiliary_px[1]
            auxiliary[2, px, py] = auxiliary_px[2]
        end
    end
end

@inbounds @inline function gaussian_power(
    con_o::SVector{4, Float32}, δxy::SVector{2, Float32},
)
    -0.5f0 * (con_o[1] * δxy[1]^2 + con_o[3] * δxy[2]^2) -
        con_o[2] * δxy[1] * δxy[2]
end

@inbounds @inline to_homogeneous(x) = SVector{4, Float32}(x[1], x[2], x[3], 1f0)

@inbounds @inline function eigvals_2D(cov::SVector{3, Float32}, det::Float32)
    mid = 0.5f0 * (cov[1] + cov[3])
    tmp = √(max(0.1f0, mid^2 - det))
    λ1 = mid + tmp
    λ2 = mid - tmp
    return λ1, λ2
end

function max_eigval_2D(Σ_2D::SMatrix{2, 2, Float32, 4}, det::Float32)
    mid = 0.5f0 * (Σ_2D[1, 1] + Σ_2D[2, 2])
    return mid + sqrt(max(0.1f0, mid * mid - det))
end

@inbounds @inline function computeCov2D(
    point::SVector{4, Float32},
    focal_xy::SVector{2, Float32},
    tan_fov_xy::SVector{2, Float32},
    resolution::SVector{2, Int32},
    principal::SVector{2, Float32},
    Σ::SVector{6, Float32},
    view::SMatrix{4, 4, Float32, 16},
    ::Val{backward} = Val{false}(),
) where backward
    pv = view * point
    pxpz, pypz = (pv[1] / pv[3]), (pv[2] / pv[3])

    scaled_tan_fov_xy = 0.3f0 .* tan_fov_xy
    scaled_principal = principal .* resolution
    lim_xy = (resolution .- scaled_principal) ./ focal_xy .+ scaled_tan_fov_xy
    lim_xy_neg = scaled_principal ./ focal_xy .+ scaled_tan_fov_xy

    t = SVector{3, Float32}(
        min(lim_xy[1], max(-lim_xy_neg[1], pxpz)) * pv[3],
        min(lim_xy[2], max(-lim_xy_neg[2], pypz)) * pv[3],
        pv[3])

    # W & J are already transposed, because T = W' * J'.
    J = SMatrix{3, 3, Float32, 9}(
        focal_xy[1] / t[3], 0f0, -(focal_xy[1] * t[1]) / t[3]^2,
        0f0, focal_xy[2] / t[3], -(focal_xy[2] * t[2]) / t[3]^2,
        0f0, 0f0, 0f0)
    W = SMatrix{3, 3, Float32, 9}(
        view[1, 1], view[1, 2], view[1, 3],
        view[2, 1], view[2, 2], view[2, 3],
        view[3, 1], view[3, 2], view[3, 3])
    # TODO
    # W is the rotation of [R|t] w2c transform
    # apply it to Vrk cov
    # https://github.com/nerfstudio-project/gsplat/blob/fc1a3ca8b901279461a8dca2676eb9d600c18b7c/gsplat/cuda/csrc/utils.cuh#L261
    T = W * J

    Vrk = SMatrix{3, 3, Float32, 9}(
        Σ[1], Σ[2], Σ[3],
        Σ[2], Σ[4], Σ[5],
        Σ[3], Σ[5], Σ[6])
    # Eq. (5).
    cov = transpose(T) * Vrk * T
    # Apply low-pass filter: every Gaussian should be at least 1-pixel wide/high.
    # Discard 3rd row and column.
    cov_sub = SVector{3, Float32}(cov[1, 1] + 0.3f0, cov[1, 2], cov[2, 2] + 0.3f0)

    if backward
        x_grad_mul = ifelse(-lim_xy_neg[1] ≤ pxpz ≤ lim_xy[1], 1f0, 0f0)
        y_grad_mul = ifelse(-lim_xy_neg[2] ≤ pypz ≤ lim_xy[2], 1f0, 0f0)
        return cov_sub, J, T, W, Vrk, t, x_grad_mul, y_grad_mul
    else
        return cov_sub
    end
end

# Convert scale and rotation properties of each Gaussian to a 3D covariance
# matrix in world space.
@inbounds @inline function computeCov3D(
    scale::SVector{3, Float32}, rotation::SVector{4, Float32},
    scale_modifier::Float32,
)
    scale = scale * scale_modifier
    # Eq. 6.
    S = sdiagm(scale...)
    R = transpose(quat2mat(rotation))
    M = S * R # M = S' * R'
    Σ = transpose(M) * M
    # Covariance is symmetric, return upper-right part.
    SVector{6, Float32}(
        Σ[1, 1], Σ[1, 2], Σ[1, 3],
        Σ[2, 2], Σ[2, 3], Σ[3, 3])
end

# Convert spherical harmonics coefficients of each Gaussian to a RGB color.
@inbounds @inline function compute_colors_from_sh(
    point::SVector{3, Float32}, camera_position::SVector{3, Float32},
    shs::AbstractVector{SVector{3, Float32}}, ::Val{degree}
) where degree
    res = SH0 * shs[1]
    if degree > 0
        dir = normalize(point - camera_position)
        x, y, z = dir
        res = res - SH1 * y * shs[2] + SH1 * z * shs[3] - SH1 * x * shs[4]
        if degree > 1
            x², y², z² = x^2, y^2, z^2
            xy, xz, yz = x * y, x * z, y * z
            res = res +
                SH2C1 * xy * shs[5] +
                SH2C2 * yz * shs[6] +
                SH2C3 * (2f0 * z² - x² - y²) * shs[7] +
                SH2C4 * xz * shs[8] +
                SH2C5 * (x² - y²) * shs[9]

            if degree > 2
                res = res +
                    SH3C1 * y * (3f0 * x² - y²) * shs[10] +
                    SH3C2 * xy * z * shs[11] +
                    SH3C3 * y * (4f0 * z² - x² - y²) * shs[12] +
                    SH3C4 * z * (2f0 * z² - 3f0 * x² - 3f0 * y²) * shs[13] +
                    SH3C5 * x * (4f0 * z² - x² - y²) * shs[14] +
                    SH3C6 * z * (x² - y²) * shs[15] +
                    SH3C7 * x * (x² - 3f0 * y²) * shs[16]
            end
        end
    end
    res = res .+ 0.5f0 .+ eps(Float32) # Add for stability.
    return max.(0f0, res), (res .< 0f0)
end
