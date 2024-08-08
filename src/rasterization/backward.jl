# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

"""
Compute loss gradients w.r.t. RGB of Gaussians, opacity, conic matrix
and 2D mean positions from per-pixel loss gradients.
"""
@kernel function ∇render!(
    # Outputs.
    ∂L∂colors::AbstractMatrix{Float32},
    ∂L∂opacities::AbstractMatrix{Float32},
    ∂L∂conic_opacities::AbstractArray{Float32, 3},
    ∂L∂means_2d::AbstractMatrix{Float32},
    # Inputs.
    ∂L∂pixels::AbstractMatrix{SVector{3, Float32}},
    # (output from the forward `render!` pass)
    n_contrib::AbstractMatrix{UInt32},
    accum_α::AbstractMatrix{Float32},

    gaussian_values_sorted::AbstractVector{UInt32},
    means_2d::AbstractVector{SVector{2, Float32}},
    conic_opacities::AbstractVector{SVector{4, Float32}},
    rgb_features::AbstractVector{SVector{3, Float32}},

    ranges::AbstractMatrix{UInt32},
    resolution::SVector{2, Int32},
    bg_color::SVector{3, Float32},
    grid::SVector{2, Int32}, block::SVector{2, Int32},
    ::Val{block_size}, ::Val{channels},
) where {block_size, channels}
    @uniform horizontal_blocks = gpu_cld(resolution[1], block[1])

    gidx = @index(Group, NTuple) # ≡ group_index
    lidx = @index(Local, NTuple) # ≡ thread_index
    ridx = @index(Local)         # ≡ thread_rank

    # Get current tile and starting pixel range.
    pix_min = SVector{2, Int32}(
        (gidx[1] - 1i32) * block[1],
        (gidx[2] - 1i32) * block[2])
    # 0-based indices.
    pix = SVector{2, Int32}(
        pix_min[1] + lidx[1] - 1i32,
        pix_min[2] + lidx[2] - 1i32)
    px, py = pix .+ 1i32 # 1-based indices.

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
    collected_colors = @localmem SVector{3, Float32} block_size
    collected_xy = @localmem SVector{2, Float32} block_size
    collected_id = @localmem UInt32 block_size

    # Start rasterization from the back.
    @inbounds T_final = inside ? accum_α[px, py] : 0f0
    T = T_final

    contributor = to_do
    @inbounds last_contributor = inside ? n_contrib[px, py] : 0i32

    accum_rec = zeros(MVector{3, Float32})
    @inbounds ∂L∂pixel = inside ? ∂L∂pixels[px, py] : zeros(SVector{3, Float32})

    last_α = 0f0
    last_color = zeros(MVector{3, Float32})

    # Gradient of pixel coordinate w.r.t. normalized
    # screen-space viewport coordinates [-1, 1].
    ∂delx∂x, ∂dely∂y = 0.5f0 * resolution[1], 0.5f0 * resolution[2]

    # Iterate over batches.
    for round in 0i32:(rounds - 1i32)
        @synchronize()
        # Load data into shared memory in reverse order from the back.
        progress = block_size * round + ridx # 1-based.
        if range[1] + progress ≤ range[2]
            # gaussian_id = gaussian_values_sorted[progress]
            @inbounds gaussian_id = gaussian_values_sorted[range[2] - progress + 1i32]
            @inbounds collected_id[ridx] = gaussian_id
            @inbounds collected_xy[ridx] = means_2d[gaussian_id]
            @inbounds collected_conic_opacity[ridx] = conic_opacities[gaussian_id]
            @inbounds collected_colors[ridx] = rgb_features[gaussian_id]
        end
        @synchronize()

        # If `done`, this thread only helps with data fetching.
        done == Cint(1) && continue

        # Iterate over current batch.
        for j in 1i32:min(block_size, to_do)
            contributor -= 1i32
            # Skip to the one behind the last.
            contributor ≥ last_contributor && continue

            # Resample using conic matrix:
            # ("Surface Splatting" by Zwicker et al., 2001).
            @inbounds xy = collected_xy[j]
            δxy = xy .- pix
            @inbounds con_o = collected_conic_opacity[j]
            power = gaussian_power(con_o, δxy)
            power > 0f0 && continue

            # Compute α.
            G = exp(power)
            α = min(0.99f0, con_o[4] * G)
            α < (1f0 / 255f0) && continue

            T /= 1f0 - α
            ∂channel∂color = α * T

            # Propagate gradients to per-Gaussian colors and keep
            # gradients w.r.t. α (blending factor).
            ∂L∂α = 0f0
            @inbounds gaussian_id = collected_id[j]
            @inbounds color = collected_colors[j]
            for c in 1i32:channels
                # Update last color (to be used in the next iteration).
                accum_rec[c] = last_α * last_color[c] + (1f0 - last_α) * accum_rec[c]
                last_color[c] = color[c]

                ∂L∂channel = ∂L∂pixel[c]
                ∂L∂α += (color[c] - accum_rec[c]) * ∂L∂channel

                # Update the gradients w.r.t. color of the Gaussian.
                # Atomically, since this pixel may be one of many, affected
                # by this Gaussian.
                @inbounds begin
                    @atomic ∂L∂colors[c, gaussian_id] += ∂channel∂color * ∂L∂channel
                end
            end

            ∂L∂α *= T
            last_α = α

            # Account for the fact that `α` also influences how
            # much of the background is added if nothing left to blend.
            ∂L∂α += (-T_final / (1f0 - α)) * (bg_color ⋅ ∂L∂pixel)

            # Temporary reusable variables.
            ∂L∂G = con_o[4] * ∂L∂α
            gdx, gdy = G * δxy[1], G * δxy[2]
            ∂G∂delx = -gdx * con_o[1] - gdy * con_o[2]
            ∂G∂dely = -gdy * con_o[3] - gdx * con_o[2]

            @inbounds begin
                # Update gradients w.r.t. 2D mean position of the Gaussian.
                @atomic ∂L∂means_2d[1, gaussian_id] += ∂L∂G * ∂G∂delx * ∂delx∂x
                @atomic ∂L∂means_2d[2, gaussian_id] += ∂L∂G * ∂G∂dely * ∂dely∂y

                # Update gradients w.r.t. 2D covariance (symmetric 2x2 matrix).
                @atomic ∂L∂conic_opacities[1, 1, gaussian_id] += -0.5f0 * gdx * δxy[1] * ∂L∂G
                @atomic ∂L∂conic_opacities[2, 1, gaussian_id] += -0.5f0 * gdx * δxy[2] * ∂L∂G
                @atomic ∂L∂conic_opacities[2, 2, gaussian_id] += -0.5f0 * gdy * δxy[2] * ∂L∂G

                # Update gradients w.r.t. opacity of tha Gaussian.
                @atomic ∂L∂opacities[gaussian_id] += G * ∂L∂α
            end
        end

        to_do -= block_size
    end
end

@kernel function ∇compute_cov_2d!(
    # Outputs.
    ∂L∂means::AbstractVector{SVector{3, Float32}},
    ∂L∂cov::AbstractMatrix{Float32},
    ∂L∂τ::Maybe{AbstractMatrix{Float32}}, # (6, N)
    # Inputs.
    ∂L∂conic_opacities::AbstractArray{Float32, 3},
    cov3Ds::AbstractVector{SVector{6, Float32}},
    radii::AbstractVector{Int32},
    means::AbstractVector{SVector{3, Float32}},
    view::SMatrix{4, 4, Float32, 16},
    focal_xy::SVector{2, Float32},
    tan_fov_xy::SVector{2, Float32},
    resolution::SVector{2, Int32},
    principal::SVector{2, Float32},
)
    i = @index(Global)
    if @inbounds(radii[i]) > 0
        @inbounds ∂L∂conic = SVector{3, Float32}( # Symmetric 2x2 matrix.
            ∂L∂conic_opacities[1, 1, i],
            ∂L∂conic_opacities[2, 1, i],
            ∂L∂conic_opacities[2, 2, i])

        @inbounds cov, J, T, W, Vrk, t, x_grad_mul, y_grad_mul = computeCov2D(
            to_homogeneous(means[i]), focal_xy, tan_fov_xy, resolution, principal,
            cov3Ds[i], view, Val{true}())

        T = transpose(T) # TODO perform transposed indexing below
        W = transpose(W)

        a, b, c = cov
        denom = a * c - b^2
        denom_inv = 1f0 / (denom^2 + eps(Float32))

        ∂L∂a, ∂L∂b, ∂L∂c = 0f0, 0f0, 0f0
        if denom_inv ≉ 0f0
            # Gradients of loss w.r.t. entries of a 2D covariance matrix
            # given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
            ∂L∂a = denom_inv * (
                -c^2 * ∂L∂conic[1] + 2f0 * b * c * ∂L∂conic[2] +
                (denom - a * c) * ∂L∂conic[3])
            ∂L∂b = denom_inv * 2f0 * (
                b * c * ∂L∂conic[1] - (denom + 2f0 * b^2) * ∂L∂conic[2] +
                a * b * ∂L∂conic[3])
            ∂L∂c = denom_inv * (
                -a^2 * ∂L∂conic[3] + 2f0 * a * b * ∂L∂conic[2] +
                (denom - a * c) * ∂L∂conic[1])

            # Gradients of loss w.r.t. each 3D covariance matrix (Vrk) entry,
            # given gradients w.r.t. 2D covariance matrix (diagonal).
            @inbounds begin
                ∂L∂cov[1, i] = (T[1,1]^2 * ∂L∂a + T[1,1] * T[2,1] * ∂L∂b + T[2,1]^2 * ∂L∂c)
                ∂L∂cov[4, i] = (T[1,2]^2 * ∂L∂a + T[1,2] * T[2,2] * ∂L∂b + T[2,2]^2 * ∂L∂c)
                ∂L∂cov[6, i] = (T[1,3]^2 * ∂L∂a + T[1,3] * T[2,3] * ∂L∂b + T[2,3]^2 * ∂L∂c)
            end

            # Gradients of loss w.r.t. each 3D covariance matrix (Vrk) entry,
            # given gradients w.r.t. 2D covariance matrix (off-diagonal).
            # Off-diagonal elements appear twice, so double the gradient.
            @inbounds ∂L∂cov[2, i] =
                2f0 * T[1, 1] * T[1, 2] * ∂L∂a +
                (T[1, 1] * T[2, 2] + T[1, 2] * T[2, 1]) * ∂L∂b +
                2f0 * T[2, 1] * T[2, 2] * ∂L∂c
            @inbounds ∂L∂cov[3, i] =
                2f0 * T[1, 1] * T[1, 3] * ∂L∂a +
                (T[1, 1] * T[2, 3] + T[1, 3] * T[2, 1]) * ∂L∂b +
                2f0 * T[2, 1] * T[2, 3] * ∂L∂c
            @inbounds ∂L∂cov[5, i] =
                2f0 * T[1, 3] * T[1, 2] * ∂L∂a +
                (T[1, 2] * T[2, 3] + T[1, 3] * T[2, 2]) * ∂L∂b +
                2f0 * T[2, 2] * T[2, 3] * ∂L∂c
        else
            @inbounds for j in 1:6
                ∂L∂cov[j, i] = 0f0
            end
        end

        # Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T.
        ∂L∂T₁₁ = 2f0 *
            ∂L∂a * (T[1,1]*Vrk[1,1] + T[1,2]*Vrk[1,2] + T[1,3]*Vrk[1,3]) +
            ∂L∂b * (T[2,1]*Vrk[1,1] + T[2,2]*Vrk[1,2] + T[2,3]*Vrk[1,3])
        ∂L∂T₁₂ = 2f0 *
            ∂L∂a * (T[1,1]*Vrk[2,1] + T[1,2]*Vrk[2,2] + T[1,3]*Vrk[2,3]) +
            ∂L∂b * (T[2,1]*Vrk[2,1] + T[2,2]*Vrk[2,2] + T[2,3]*Vrk[2,3])
        ∂L∂T₁₃ = 2f0 *
            ∂L∂a * (T[1,1]*Vrk[3,1] + T[1,2]*Vrk[3,2] + T[1,3]*Vrk[3,3]) +
            ∂L∂b * (T[2,1]*Vrk[3,1] + T[2,2]*Vrk[3,2] + T[2,3]*Vrk[3,3])
        ∂L∂T₂₁ = 2f0 *
            ∂L∂c * (T[2,1]*Vrk[1,1] + T[2,2]*Vrk[1,2] + T[2,3]*Vrk[1,3]) +
            ∂L∂b * (T[1,1]*Vrk[1,1] + T[1,2]*Vrk[1,2] + T[1,3]*Vrk[1,3])
        ∂L∂T₂₂ = 2f0 *
            ∂L∂c * (T[2,1]*Vrk[2,1] + T[2,2]*Vrk[2,2] + T[2,3]*Vrk[2,3]) +
            ∂L∂b * (T[1,1]*Vrk[2,1] + T[1,2]*Vrk[2,2] + T[1,3]*Vrk[2,3])
        ∂L∂T₂₃ = 2f0 *
            ∂L∂c * (T[2,1]*Vrk[3,1] + T[2,2]*Vrk[3,2] + T[2,3]*Vrk[3,3]) +
            ∂L∂b * (T[1,1]*Vrk[3,1] + T[1,2]*Vrk[3,2] + T[1,3]*Vrk[3,3])

        # Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix.
        ∂L∂J₁₁ = W[1, 1] * ∂L∂T₁₁ + W[1, 2] * ∂L∂T₁₂ + W[1, 3] * ∂L∂T₁₃
        ∂L∂J₁₃ = W[3, 1] * ∂L∂T₁₁ + W[3, 2] * ∂L∂T₁₂ + W[3, 3] * ∂L∂T₁₃
        ∂L∂J₂₂ = W[2, 1] * ∂L∂T₂₁ + W[2, 2] * ∂L∂T₂₂ + W[2, 3] * ∂L∂T₂₃
        ∂L∂J₂₃ = W[3, 1] * ∂L∂T₂₁ + W[3, 2] * ∂L∂T₂₂ + W[3, 3] * ∂L∂T₂₃

        tz = 1f0 / t[3]
        tz² = tz^2
        tz³ = tz² * tz

        # Gradients of loss w.r.t. transformed Gaussian mean t.
        ∂L∂t = SVector{3, Float32}(
            x_grad_mul * -focal_xy[1] * tz² * ∂L∂J₁₃, # x
            y_grad_mul * -focal_xy[2] * tz² * ∂L∂J₂₃, # y
            -focal_xy[1] * tz² * ∂L∂J₁₁ +             # z
                -focal_xy[2] * tz² * ∂L∂J₂₂ +
                (2f0 * focal_xy[1] * t[1]) * tz³ * ∂L∂J₁₃ +
                (2f0 * focal_xy[2] * t[2]) * tz³ * ∂L∂J₂₃)

        # Account for transformation of mean to t (apply only inv rotation).
        rot_inv = SMatrix{3, 3, Float32, 9}(
            view[1, 1], view[1, 2], view[1, 3],
            view[2, 1], view[2, 2], view[2, 3],
            view[3, 1], view[3, 2], view[3, 3])
        @inbounds ∂L∂means[i] = rot_inv * ∂L∂t

        # Compute camera gradients.
        # TODO inbounds
        if !isnothing(∂L∂τ)
            ∂pC_∂ρ = SMatrix{3, 3, Float32}(I)
            ∂pC_∂θ = skew_sym_mat(-t)

            ∂τ = MVector{6, Float32}(undef)
            for j in 1:3
                ∂τ[j] =
                    ∂L∂t[1] * ∂pC_∂ρ[1, j] +
                    ∂L∂t[2] * ∂pC_∂ρ[2, j] +
                    ∂L∂t[3] * ∂pC_∂ρ[3, j]
                ∂τ[j + 3] =
                    ∂L∂t[1] * ∂pC_∂θ[1, j] +
                    ∂L∂t[2] * ∂pC_∂θ[2, j] +
                    ∂L∂t[3] * ∂pC_∂θ[3, j]
            end
            for j in 1:6
                # NOTE: without @inbounds we get LLVM IR error: gc_frame...
                @inbounds ∂L∂τ[j, i] += ∂τ[j]
            end

            ∂L∂W₁₁ = J[1, 1] * ∂L∂T₁₁
            ∂L∂W₁₂ = J[1, 1] * ∂L∂T₁₂
            ∂L∂W₁₃ = J[1, 1] * ∂L∂T₁₃

            ∂L∂W₂₁ = J[2, 2] * ∂L∂T₂₁
            ∂L∂W₂₂ = J[2, 2] * ∂L∂T₂₂
            ∂L∂W₂₃ = J[2, 2] * ∂L∂T₂₃

            ∂L∂W₃₁ = J[1, 3] * ∂L∂T₁₁ + J[2, 3] * ∂L∂T₂₁
            ∂L∂W₃₂ = J[1, 3] * ∂L∂T₁₂ + J[2, 3] * ∂L∂T₂₂
            ∂L∂W₃₃ = J[1, 3] * ∂L∂T₁₃ + J[2, 3] * ∂L∂T₂₃

            ∂L∂W = SMatrix{3, 3, Float32, 9}(
                ∂L∂W₁₁, ∂L∂W₂₁, ∂L∂W₃₁,
                ∂L∂W₁₂, ∂L∂W₂₂, ∂L∂W₃₂,
                ∂L∂W₁₃, ∂L∂W₂₃, ∂L∂W₃₃)

            n_W1_x = skew_sym_mat(-view[:, 1])
            n_W2_x = skew_sym_mat(-view[:, 2])
            n_W3_x = skew_sym_mat(-view[:, 3])

            ∂L∂θ = SVector{3, Float32}(
                ∂L∂W[:, 1] ⋅ n_W1_x[:, 1] +
                ∂L∂W[:, 2] ⋅ n_W2_x[:, 1] +
                ∂L∂W[:, 3] ⋅ n_W3_x[:, 1],

                ∂L∂W[:, 1] ⋅ n_W1_x[:, 2] +
                ∂L∂W[:, 2] ⋅ n_W2_x[:, 2] +
                ∂L∂W[:, 3] ⋅ n_W3_x[:, 2],

                ∂L∂W[:, 1] ⋅ n_W1_x[:, 3] +
                ∂L∂W[:, 2] ⋅ n_W2_x[:, 3] +
                ∂L∂W[:, 3] ⋅ n_W3_x[:, 3])

            ∂L∂τ[4, i] += ∂L∂θ[1]
            ∂L∂τ[5, i] += ∂L∂θ[2]
            ∂L∂τ[6, i] += ∂L∂θ[3]
        end
    end
end

"""
Backward pass of the preprocessing steps, except
for the covariance computation and inversion,
"""
@kernel function ∇_preprocess!(
    # Outputs.
    ∂L∂means::AbstractVector{SVector{3, Float32}},
    ∂L∂shs::AbstractMatrix{SVector{3, Float32}},
    ∂L∂scales::AbstractVector{SVector{3, Float32}},
    ∂L∂rot::AbstractVector{SVector{4, Float32}},
    ∂L∂τ::Maybe{AbstractMatrix{Float32}}, # (6, N), [ρ, θ]
    # Inputs.
    ∂L∂cov::AbstractMatrix{Float32},
    ∂L∂colors::AbstractVector{SVector{3, Float32}},
    ∂L∂means_2d::AbstractVector{SVector{2, Float32}},
    radii::AbstractVector{Int32},
    means::AbstractVector{SVector{3, Float32}},
    scales::AbstractVector{SVector{3, Float32}}, # For cov 3D
    rotations::AbstractVector{SVector{4, Float32}}, # For cov 3D
    spherical_harmonics::AbstractMatrix{SVector{3, Float32}},
    sh_degree,
    clamped::AbstractVector{SVector{3, Bool}},
    projection::SMatrix{4, 4, Float32, 16},
    projection_raw::SMatrix{4, 4, Float32, 16},
    view::SMatrix{4, 4, Float32, 16},
    camera_position::SVector{3, Float32},
    scale_modifier::Float32,
)
    i = @index(Global)
    if @inbounds(radii[i] > 0)
        @inbounds point = means[i]
        point_h = to_homogeneous(point)

        # Project point into camera space.
        projected_h = projection * point_h
        pw = 1f0 / (projected_h[4] + eps(Float32))

        # Loss gradient w.r.t. 3D means due to gradients of 2D means
        # from forward pass.
        mult_1 = pw^2 * (
            projection[1, 1] * point[1] + projection[1, 2] * point[2] +
            projection[1, 3] * point[3] + projection[1, 4])
        mult_2 = pw^2 *(
            projection[2, 1] * point[1] + projection[2, 2] * point[2] +
            projection[2, 3] * point[3] + projection[2, 4])

        @inbounds ∂L∂mean_2d = ∂L∂means_2d[i]
        ∂L∂mean = SVector{3, Float32}(
            # x
            (projection[1, 1] * pw - projection[4, 1] * mult_1) * ∂L∂mean_2d[1] +
            (projection[2, 1] * pw - projection[4, 1] * mult_2) * ∂L∂mean_2d[2],
            # y
            (projection[1, 2] * pw - projection[4, 2] * mult_1) * ∂L∂mean_2d[1] +
            (projection[2, 2] * pw - projection[4, 2] * mult_2) * ∂L∂mean_2d[2],
            # z
            (projection[1, 3] * pw - projection[4, 3] * mult_1) * ∂L∂mean_2d[1] +
            (projection[2, 3] * pw - projection[4, 3] * mult_2) * ∂L∂mean_2d[2])

        ∂L∂mean_sh = ∇color_from_sh!(
            @view(∂L∂shs[:, i]), # Output.
            point, camera_position, @view(spherical_harmonics[:, i]), # Inputs.
            sh_degree, clamped[i], ∂L∂colors[i])
        @inbounds ∂L∂means[i] += ∂L∂mean + ∂L∂mean_sh

        if !isnothing(∂L∂τ)
            # TODO inbounds
            α = pw
            β = -projected_h[1] * α^2
            γ = -projected_h[2] * α^2

            a = projection_raw[1, 1]
            b = projection_raw[2, 2]
            c = projection_raw[3, 3]
            d = projection_raw[3, 4] # TODO swap d & e?
            e = projection_raw[4, 3]

            pC = view * point_h
            ∂pC_∂ρ = SMatrix{3, 3, Float32}(I)
            ∂pC_∂θ = skew_sym_mat(-pC)

            ∂proj∂pC1 = SVector{3, Float32}(α * a, 0f0, β * e)
            ∂proj∂pC2 = SVector{3, Float32}(0f0, α * b, γ * e)

            ∂proj∂pC1_∂ρ = ∂pC_∂ρ * ∂proj∂pC1
            ∂proj∂pC2_∂ρ = ∂pC_∂ρ * ∂proj∂pC2
            ∂proj∂pC1_∂θ = transpose(∂pC_∂θ) * ∂proj∂pC1
            ∂proj∂pC2_∂θ = transpose(∂pC_∂θ) * ∂proj∂pC2

            ∂mean_2d∂τ = MMatrix{2, 6, Float32}(undef)
            ∂mean_2d∂τ[1, 1] = ∂proj∂pC1_∂ρ[1]
            ∂mean_2d∂τ[1, 2] = ∂proj∂pC1_∂ρ[2]
            ∂mean_2d∂τ[1, 3] = ∂proj∂pC1_∂ρ[3]
            ∂mean_2d∂τ[1, 4] = ∂proj∂pC1_∂θ[1]
            ∂mean_2d∂τ[1, 5] = ∂proj∂pC1_∂θ[2]
            ∂mean_2d∂τ[1, 6] = ∂proj∂pC1_∂θ[3]

            ∂mean_2d∂τ[2, 1] = ∂proj∂pC2_∂ρ[1]
            ∂mean_2d∂τ[2, 2] = ∂proj∂pC2_∂ρ[2]
            ∂mean_2d∂τ[2, 3] = ∂proj∂pC2_∂ρ[3]
            ∂mean_2d∂τ[2, 4] = ∂proj∂pC2_∂θ[1]
            ∂mean_2d∂τ[2, 5] = ∂proj∂pC2_∂θ[2]
            ∂mean_2d∂τ[2, 6] = ∂proj∂pC2_∂θ[3]

            # NOTE: without @inbounds we get LLVM IR error: gc_frame...
            @inbounds for j in 1:6
                ∂L∂τ[j, i] +=
                    ∂L∂mean_2d[1] * ∂mean_2d∂τ[1, j] +
                    ∂L∂mean_2d[2] * ∂mean_2d∂τ[2, j]
            end

            ∂L∂τ[1, i] -= ∂L∂mean_sh[1]
            ∂L∂τ[2, i] -= ∂L∂mean_sh[2]
            ∂L∂τ[3, i] -= ∂L∂mean_sh[3]
        end

        @inbounds ∂L∂scale, ∂L∂q = ∇compute_cov_3d(
            @view(∂L∂cov[:, i]), scales[i], rotations[i], scale_modifier)
        @inbounds ∂L∂scales[i] = ∂L∂scale
        @inbounds ∂L∂rot[i] = ∂L∂q
    end
end

function ∇color_from_sh!(
    # Outputs.
    ∂L∂shs::AbstractVector{SVector{3, Float32}},
    # Inputs.
    point::SVector{3, Float32},
    camera_position::SVector{3, Float32},
    shs::AbstractVector{SVector{3, Float32}}, ::Val{degree},
    clamped::SVector{3, Bool},
    ∂L∂color::SVector{3, Float32},
) where degree
    dir_orig = point - camera_position
    dir = normalize(dir_orig)

    # If clamped - gradient is 0.
    ∂L∂color = ∂L∂color .* (1f0 .- clamped)
    ∂color∂x = zeros(SVector{3, Float32})
    ∂color∂y = zeros(SVector{3, Float32})
    ∂color∂z = zeros(SVector{3, Float32})

    @inbounds ∂L∂shs[1] = SH0 * ∂L∂color
    @inbounds if degree > 0
        x, y, z = dir
        ∂L∂shs[2] = -SH1 * y * ∂L∂color
        ∂L∂shs[3] =  SH1 * z * ∂L∂color
        ∂L∂shs[4] = -SH1 * x * ∂L∂color

        ∂color∂x = -SH1 * shs[4]
        ∂color∂y = -SH1 * shs[2]
        ∂color∂z =  SH1 * shs[3]
        @inbounds if degree > 1
            x², y², z² = x^2, y^2, z^2
            xy, xz, yz = x * y, x * z, y * z

            ∂L∂shs[5] = SH2C1 * xy * ∂L∂color
            ∂L∂shs[6] = SH2C2 * yz * ∂L∂color
            ∂L∂shs[7] = SH2C3 * (2f0 * z² - x² - y²) * ∂L∂color
            ∂L∂shs[8] = SH2C4 * xz * ∂L∂color
            ∂L∂shs[9] = SH2C5 * (x² - y²) * ∂L∂color

            ∂color∂x = ∂color∂x +
                SH2C1 * y * shs[5] +
                SH2C3 * 2f0 * -x * shs[7] +
                SH2C4 * z * shs[8] +
                SH2C5 * 2f0 * x * shs[9]
            ∂color∂y = ∂color∂y +
                SH2C1 * x * shs[5] +
                SH2C2 * z * shs[6] +
                SH2C3 * 2f0 * -y * shs[7] +
                SH2C5 * 2f0 * -y * shs[9]
            ∂color∂z = ∂color∂z +
                SH2C2 * y * shs[6] +
                SH2C3 * 4f0 * z * shs[7] +
                SH2C4 * x * shs[8]
            @inbounds if degree > 2
                ∂L∂shs[10] = SH3C1 * y * (3f0 * x² - y²) * ∂L∂color
                ∂L∂shs[11] = SH3C2 * xy * z * ∂L∂color
                ∂L∂shs[12] = SH3C3 * y * (4f0 * z² - x² - y²) * ∂L∂color
                ∂L∂shs[13] = SH3C4 * z * (2f0 * z² - 3f0 * x² - 3f0 * y²) * ∂L∂color
                ∂L∂shs[14] = SH3C5 * x * (4f0 * z² - x² - y²) * ∂L∂color
                ∂L∂shs[15] = SH3C6 * z * (x² - y²) * ∂L∂color
                ∂L∂shs[16] = SH3C7 * x * (x² - 3f0 * y²) * ∂L∂color

                ∂color∂x = ∂color∂x +
                    SH3C1 * shs[10] * 3f0 * 2f0 * xy +
                    SH3C2 * shs[11] * yz +
                    SH3C3 * shs[12] * -2f0 * xy +
                    SH3C4 * shs[13] * -3f0 * 2f0 * xz +
                    SH3C5 * shs[14] * (-3f0 * x² + 4f0 * z² - y²) +
                    SH3C6 * shs[15] * 2f0 * xz +
                    SH3C7 * shs[16] * 3f0 * (x² - y²)
                ∂color∂y = ∂color∂y +
                    SH3C1 * shs[10] * 3f0 * (x² - y²) +
                    SH3C2 * shs[11] * xz +
                    SH3C3 * shs[12] * (-3f0 * y² + 4f0 * z² - x²) +
                    SH3C4 * shs[13] * -3f0 * 2f0 * yz +
                    SH3C5 * shs[14] * -2f0 * xy +
                    SH3C6 * shs[15] * -2f0 * yz +
                    SH3C7 * shs[16] * -3f0 * 2f0 * xy
                ∂color∂z = ∂color∂z +
                    SH3C2 * shs[11] * xy +
                    SH3C3 * shs[12] * 4f0 * 2f0 * yz +
                    SH3C4 * shs[13] * 3f0 * (2f0 * z² - x² - y²) +
                    SH3C5 * shs[14] * 4f0 * 2f0 * xz +
                    SH3C6 * shs[15] * (x² - y²)
            end
        end
    end

    # The view direction is an input to the computation.
    # View direction is influenced by the Gaussian's mean,
    # so SHs gradients must propagate back into 3D position.
    ∂L∂dir = SVector{3, Float32}(
        ∂color∂x ⋅ ∂L∂color, ∂color∂y ⋅ ∂L∂color, ∂color∂z ⋅ ∂L∂color)

    # Account for normalization.
    return ∇normalize(dir_orig, ∂L∂dir)
end

function ∇normalize(dir::SVector{3, Float32}, ∂L∂dir::SVector{3, Float32})
    s² = sum(abs2, dir)
    inv_s = 1f0 / √(s²^3)
    SVector{3, Float32}(
        ((s² - dir[1]^2) * ∂L∂dir[1] - dir[2] * dir[1] * ∂L∂dir[2] - dir[3] * dir[1] * ∂L∂dir[3]) * inv_s,
        (-dir[1] * dir[2] * ∂L∂dir[1] + (s² - dir[2]^2) * ∂L∂dir[2] - dir[3] * dir[2] * ∂L∂dir[3]) * inv_s,
        (-dir[1] * dir[3] * ∂L∂dir[1] - dir[2] * dir[3] * ∂L∂dir[2] + (s² - dir[3]^2) * ∂L∂dir[3]) * inv_s)
end

function ∇compute_cov_3d(
    ∂L∂cov::AbstractVector{Float32},
    scale::SVector{3, Float32}, rotation::SVector{4, Float32},
    scale_modifier::Float32,
)
    scale = scale * scale_modifier
    S = sdiagm(scale...)
    R = transpose(quat2mat(rotation))
    M = S * R # M = S' * R'

    @inbounds begin
        dunc = SVector{3, Float32}(∂L∂cov[1], ∂L∂cov[4], ∂L∂cov[6])
        ounc = SVector{3, Float32}(∂L∂cov[2], ∂L∂cov[3], ∂L∂cov[5]) .* 0.5f0
        ∂L∂Σ = SMatrix{3, 3, Float32, 9}(
            ∂L∂cov[1], 0.5f0 * ∂L∂cov[2], 0.5f0 * ∂L∂cov[3],
            0.5f0 * ∂L∂cov[2], ∂L∂cov[4], 0.5f0 * ∂L∂cov[5],
            0.5f0 * ∂L∂cov[3], 0.5f0 * ∂L∂cov[5], ∂L∂cov[6])
    end

    # Compute loss gradient w.r.t. matrix M.
    ∂L∂M = 2f0 * M * ∂L∂Σ
    Rᵗ = transpose(R)
    ∂L∂Mᵗ = transpose(∂L∂M)

    # Loss gradient w.r.t. scale.
    ∂L∂scale = SVector{3, Float32}(
        @view(Rᵗ[1, :]) ⋅ @view(∂L∂Mᵗ[1, :]),
        @view(Rᵗ[2, :]) ⋅ @view(∂L∂Mᵗ[2, :]),
        @view(Rᵗ[3, :]) ⋅ @view(∂L∂Mᵗ[3, :]))

    ∂L∂Mᵗ = SMatrix{3, 3, Float32, 9}(
        ∂L∂Mᵗ[1, 1] * scale[1], ∂L∂Mᵗ[2, 1] * scale[2], ∂L∂Mᵗ[3, 1] * scale[3],
        ∂L∂Mᵗ[1, 2] * scale[1], ∂L∂Mᵗ[2, 2] * scale[2], ∂L∂Mᵗ[3, 2] * scale[3],
        ∂L∂Mᵗ[1, 3] * scale[1], ∂L∂Mᵗ[2, 3] * scale[2], ∂L∂Mᵗ[3, 3] * scale[3])

    # Loss gradient w.r.t. normalized quaternion.
    r, x, y, z = rotation
    ∂L∂rot = SVector{4, Float32}(
        # r
        (2f0 * z * ∂L∂Mᵗ[1, 2] - 2f0 * z * ∂L∂Mᵗ[2, 1]) +
        (2f0 * y * ∂L∂Mᵗ[3, 1] - 2f0 * y * ∂L∂Mᵗ[1, 3]) +
        (2f0 * x * ∂L∂Mᵗ[2, 3] - 2f0 * x * ∂L∂Mᵗ[3, 2]),
        # x
        (2f0 * y * ∂L∂Mᵗ[2, 1] + 2f0 * y * ∂L∂Mᵗ[1, 2]) +
        (2f0 * z * ∂L∂Mᵗ[3, 1] + 2f0 * z * ∂L∂Mᵗ[1, 3]) +
        (2f0 * r * ∂L∂Mᵗ[2, 3] - 2f0 * r * ∂L∂Mᵗ[3, 2]) -
        (4f0 * x * ∂L∂Mᵗ[3, 3] + 4f0 * x * ∂L∂Mᵗ[2, 2]),
        # y
        (2f0 * x * ∂L∂Mᵗ[2, 1] + 2f0 * x * ∂L∂Mᵗ[1, 2]) +
        (2f0 * r * ∂L∂Mᵗ[3, 1] - 2f0 * r * ∂L∂Mᵗ[1, 3]) +
        (2f0 * z * ∂L∂Mᵗ[2, 3] + 2f0 * z * ∂L∂Mᵗ[3, 2]) -
        (4f0 * y * ∂L∂Mᵗ[3, 3] + 4f0 * y * ∂L∂Mᵗ[1, 1]),
        # z
        (2f0 * r * ∂L∂Mᵗ[1, 2] - 2f0 * r * ∂L∂Mᵗ[2, 1]) +
        (2f0 * x * ∂L∂Mᵗ[3, 1] + 2f0 * x * ∂L∂Mᵗ[1, 3]) +
        (2f0 * y * ∂L∂Mᵗ[2, 3] + 2f0 * y * ∂L∂Mᵗ[3, 2]) -
        (4f0 * z * ∂L∂Mᵗ[2, 2] + 4f0 * z * ∂L∂Mᵗ[1, 1]))

    return ∂L∂scale, ∂L∂rot
end
