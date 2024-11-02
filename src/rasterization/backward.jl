@kernel cpu=false inbounds=true function ∇render!(
    # Outputs.
    vcolors::AbstractMatrix{Float32},
    vopacities::AbstractMatrix{Float32},
    vconics::AbstractMatrix{Float32},
    vmeans_2d::AbstractMatrix{Float32},
    # Inputs.
    vpixels::AbstractMatrix{SVector{channels, Float32}},
    n_contrib::AbstractMatrix{UInt32},
    accum_α::AbstractMatrix{Float32},

    gaussian_values_sorted::AbstractVector{UInt32},
    means_2d::AbstractVector{SVector{2, Float32}},
    opacities::AbstractMatrix{Float32},
    conics::AbstractVector{SVector{3, Float32}},
    rgb_features::AbstractVector{SVector{channels, Float32}},

    ranges::AbstractMatrix{UInt32},
    resolution::SVector{2, Int32},
    bg_color::SVector{channels, Float32},
    grid::SVector{2, Int32}, block::SVector{2, Int32}, ::Val{block_size},
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
    done::Bool = ifelse(inside, false, true)

    # Load start/end range of IDs to process.
    range_idx = (gidx[2] - 1i32) * horizontal_blocks + gidx[1]
    range = (Int32(ranges[1, range_idx]), Int32(ranges[2, range_idx]))
    to_do::Int32 = range[2] - range[1]
    # If `to_do` > `block_size`, repeat rasterization several times
    # with workitems in the workgroup.
    rounds::Int32 = gpu_cld(to_do, block_size)

    # Allocate storage for batches of collectively fetched data.
    collected_conics = @localmem SVector{3, Float32} block_size
    collected_colors = @localmem SVector{channels, Float32} block_size
    collected_xy = @localmem SVector{2, Float32} block_size
    collected_opacity = @localmem Float32 block_size
    collected_id = @localmem UInt32 block_size

    # Start rasterization from the back.
    T_final = inside ? accum_α[px, py] : 0f0
    T = T_final

    contributor = to_do
    last_contributor = inside ? n_contrib[px, py] : 0i32

    accum_rec = zeros(MVector{channels, Float32})
    vpixel = inside ? vpixels[px, py] : zeros(SVector{channels, Float32})
    last_color = zeros(MVector{channels, Float32})
    last_α = 0f0

    for round in 0i32:(rounds - 1i32)
        @synchronize()
        # Load data into shared memory in reverse order from the back.
        progress = block_size * round + ridx # 1-based.
        if range[1] + progress ≤ range[2]
            # gaussian_id = gaussian_values_sorted[progress]
            gaussian_id = gaussian_values_sorted[range[2] - progress + 1i32]
            collected_id[ridx] = gaussian_id
            collected_xy[ridx] = means_2d[gaussian_id]
            collected_opacity[ridx] = opacities[gaussian_id]
            collected_conics[ridx] = conics[gaussian_id]
            collected_colors[ridx] = rgb_features[gaussian_id]
        end
        @synchronize()

        # If `done`, this thread only helps with data fetching.
        done && continue

        for j in 1i32:min(block_size, to_do)
            contributor -= 1i32
            # Skip to the one behind the last.
            contributor ≥ last_contributor && continue

            xy = collected_xy[j]
            δ = xy .- pix
            opacity = collected_opacity[j]
            conic = collected_conics[j]
            σ = conic[2] * δ[1] * δ[2] +
                0.5f0 * (conic[1] * δ[1]^2 + conic[3] * δ[2]^2)
            σ < 0f0 && continue # TODO replace with `valid` flag and if/else to avoid divergence?

            G = exp(-σ)
            α = min(0.99f0, opacity * G)
            α < (1f0 / 255f0) && continue

            T /= 1f0 - α
            fac = α * T

            gaussian_id = collected_id[j]
            @unroll for c in 1i32:channels
                @atomic vcolors[c, gaussian_id] += fac * vpixel[c]
            end

            vα = 0f0
            color = collected_colors[j]
            @unroll for c in 1i32:channels
                # Update last color (to be used in the next iteration).
                accum_rec[c] = last_α * last_color[c] + (1f0 - last_α) * accum_rec[c]
                last_color[c] = color[c]
                vα += (color[c] - accum_rec[c]) * vpixel[c]
            end
            vα *= T
            # Account for the fact that `α` also influences how
            # much of the background is added.
            vα += (-T_final / (1f0 - α)) * (bg_color ⋅ vpixel)

            last_α = α

            vσ = -opacity * G * vα
            vconic = SVector{3, Float32}(
                0.5f0 * vσ * δ[1]^2,
                0.5f0 * vσ * δ[1] * δ[2],
                0.5f0 * vσ * δ[2]^2,
            )
            vxy = SVector{2, Float32}(
                vσ * (conic[1] * δ[1] + conic[2] * δ[2]),
                vσ * (conic[2] * δ[1] + conic[3] * δ[2]),
            )
            vopacity = G * vα

            @atomic vmeans_2d[1, gaussian_id] += vxy[1]
            @atomic vmeans_2d[2, gaussian_id] += vxy[2]

            @atomic vconics[1, gaussian_id] += vconic[1]
            @atomic vconics[2, gaussian_id] += vconic[2]
            @atomic vconics[3, gaussian_id] += vconic[3]

            @atomic vopacities[gaussian_id] += vopacity
        end
        to_do -= block_size
    end
end

@kernel cpu=false inbounds=true function ∇project!(
    # Output.
    vmeans::AbstractVector{SVector{3, Float32}},
    vcov_scales::AbstractVector{SVector{3, Float32}},
    vcov_rotations::AbstractVector{SVector{4, Float32}},

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
    R_w2c::SMatrix{3, 3, Float32, 9},
    t_w2c::SVector{3, Float32},
    focal::SVector{2, Float32},
    resolution::SVector{2, Int32},
    principal::SVector{2, Float32},
    ϵ::Float32,
) where {C <: Maybe{AbstractMatrix{Float32}}, VC, VD}
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

    mean = means[i]
    mean_cam = pos_world_to_cam(R_w2c, t_w2c, mean)

    cov_rotation, cov_scale = cov_rotations[i], cov_scales[i]
    Σ = quat_scale_to_cov(cov_rotation, cov_scale)
    Σ_cam = covar_world_to_cam(R_w2c, Σ)

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

    vR = zeros(SMatrix{3, 3, Float32, 9})
    vt = zeros(SVector{3, Float32})
    vmean = zeros(SVector{3, Float32})
    vR, vt, vmean = ∇pos_world_to_cam(
        R_w2c, t_w2c, mean,
        vmean_cam, vR, vt, vmean)

    vΣ = zeros(SMatrix{3, 3, Float32, 9})
    vR, vΣ = ∇covar_world_to_cam(R_w2c, Σ, vΣ_cam, vR, vΣ)

    vq, vscale = ∇quat_scale_to_cov(
        cov_rotation, cov_scale, unnorm_quat2rot(cov_rotation), vΣ)

    vmeans[i] = vmean
    vcov_scales[i] = vscale
    vcov_rotations[i] = vq

    # TODO write grad for vR & vt (for diff camera pose)
end
