@kernel unsafe_indices=true cpu=false inbounds=true function render!(
    # Output.
    out_color::AbstractArray{Float32, 3},
    n_contrib::AbstractMatrix{UInt32},
    accum_α::AbstractMatrix{Float32},

    covisibilities::C,
    uncertainties::U,
    # Per-gaussian edge-aware score, `Σₚ ωₚ·αₚ·Tₚ` (see `ImprovedGSStrategy`).
    # Accumulated only when both `edge_scores` & `edge_map` are given.
    edge_scores::S,
    # Input.
    edge_map::E, # Per-pixel edge weight `ω`, `(width, height)`.
    gaussian_values_sorted::AbstractVector{UInt32},
    means_2d::AbstractVector{SVector{2, Float32}},
    opacities::AbstractMatrix{Float32},
    conics::AbstractVector{SVector{3, Float32}},
    rgb_features::AbstractVector{SVector{channels, Float32}},

    ranges::AbstractMatrix{UInt32},
    resolution::SVector{2, Int32},
    background::SVector{channels, Float32},
    block::SVector{2, Int32},
    ::Val{block_size},
) where {
    block_size, channels,
    C <: Maybe{AbstractVector{Bool}},
    U <: Maybe{AbstractMatrix{Float32}},
    S <: Maybe{AbstractVector{Float32}},
    E <: Maybe{AbstractMatrix{Float32}},
}
    gidx = @index(Group, NTuple) # ≡ group_index
    lidx = @index(Local, NTuple) # ≡ thread_index
    ridx = @index(Local)         # ≡ thread_rank

    # Get current tile and starting pixel range (0-based indices).
    # 0-based indices.
    pix_min = SVector{2, Int32}(
        (gidx[1] - 1i32) * block[1],
        (gidx[2] - 1i32) * block[2])
    pix = SVector{2, Int32}(
        pix_min[1] + lidx[1] - 1i32,
        pix_min[2] + lidx[2] - 1i32)

    # Check if this thread corresponds to a valid pixel or is outside.
    # If not inside, this thread will help with data fetching,
    # but will not participate in rasterization.
    inside = pix[1] < resolution[1] && pix[2] < resolution[2]
    done::Bool = ifelse(inside, false, true)

    # Load start/end range of IDs to process.
    horizontal_blocks = gpu_cld(resolution[1], block[1])
    range_idx = (gidx[2] - 1i32) * horizontal_blocks + gidx[1]
    range = (Int32(ranges[1, range_idx]), Int32(ranges[2, range_idx]))
    to_do::Int32 = range[2] - range[1]
    # If `to_do` > `block_size`, repeat rasterization several times
    # with workitems in the workgroup.
    rounds::Int32 = gpu_cld(to_do, block_size)

    # Allocate storage for batches of collectively fetched data.
    collected_conics = @localmem SVector{3, Float32} block_size
    collected_xy = @localmem SVector{2, Float32} block_size
    collected_opacity = @localmem Float32 block_size
    collected_id = @localmem UInt32 block_size

    T = 1f0
    contributor = 0u32
    last_contributor = 0u32

    color = zeros(SVector{channels, Float32})
    uncertainty = 0f0 # Computed if `uncertainties ≢ nothing`.
    # This pixel's edge weight, hoisted out of the blend loop.
    ω = (S !== Nothing && E !== Nothing && inside) ?
        edge_map[pix[1] + 1i32, pix[2] + 1i32] : 0f0

    for round in 0i32:(rounds - 1i32)
        # Collectively fetch data from global to shared memory.
        progress = range[1] + block_size * round + ridx # 1-based.

        @synchronize()
        if progress ≤ range[2]
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
            color = color .+ feature .* (α * T)

            U !== Nothing && (uncertainty += α * T)
            # Since we are processing Gaussians front-to-back,
            # mark visible only until cummulative opacity is > 0.5.
            C !== Nothing && T > 0.5f0 && (covisibilities[gaussian_id] = true)

            if S !== Nothing && E !== Nothing
                @atomic edge_scores[gaussian_id] += ω * α * T
            end

            # Keep track of last range entry to update this pixel.
            T = T_tmp
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
        U !== Nothing && (uncertainties[px, py] = uncertainty)
    end
end

@kernel unsafe_indices=true cpu=false inbounds=true function ∇render!(
    # Outputs.
    vcolors::AbstractMatrix{Float32},
    vopacities::AbstractMatrix{Float32},
    vconics::AbstractMatrix{Float32},
    vmeans_2d::AbstractMatrix{Float32},
    # Component-wise `Σₚ |∂L/∂μ|`, accumulated only when given.
    vmeans_2d_abs::A,
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
) where {block_size, channels, A <: Maybe{AbstractMatrix{Float32}}}
    gidx = @index(Group, NTuple) # ≡ group_index
    lidx = @index(Local, NTuple) # ≡ thread_index
    ridx = @index(Local)         # ≡ thread_rank

    # Get current tile and starting pixel range (0-based indices).
    # 0-based indices.
    pix_min = SVector{2, Int32}(
        (gidx[1] - 1i32) * block[1],
        (gidx[2] - 1i32) * block[2])
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
    horizontal_blocks = gpu_cld(resolution[1], block[1])
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

    accum_rec = zeros(SVector{channels, Float32})
    vpixel = inside ? vpixels[px, py] : zeros(SVector{channels, Float32})
    last_color = zeros(SVector{channels, Float32})
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

            color = collected_colors[j]
            # Update last color (to be used in the next iteration).
            accum_rec = last_α .* last_color .+ (1f0 - last_α) .* accum_rec
            last_color = color
            vα = (color .- accum_rec) ⋅ vpixel
            # The alpha-map cotangent needs no special handling here: the map
            # is rendered as a constant-1 feature channel, so it enters `vα`
            # through the generic per-channel loop above.
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

            if A !== Nothing
                @atomic vmeans_2d_abs[1, gaussian_id] += abs(vxy[1])
                @atomic vmeans_2d_abs[2, gaussian_id] += abs(vxy[2])
            end

            @atomic vconics[1, gaussian_id] += vconic[1]
            @atomic vconics[2, gaussian_id] += vconic[2]
            @atomic vconics[3, gaussian_id] += vconic[3]

            @atomic vopacities[gaussian_id] += vopacity
        end
        to_do -= block_size
    end
end


"""
Per-splat blend backward: one workgroup per tile, **one lane per splat**, versus
`∇render!`'s one thread per pixel.

`∇render!` issues `channels + 6` atomics for every contributing (pixel, Gaussian)
pair, all contending on the same address across the workgroup. Here each lane
owns a Gaussian for the whole tile, accumulates in registers, and flushes **one**
set of atomics per (tile, splat) — `O(instances)` instead of `O(fragments)`.

The blend is serial back-to-front per pixel, so lanes cannot simply run free. A
skewed **diagonal wavefront** restores the order: at step `d`, lane `l` visits
pixel `d - l + 1`, so lane `l` only reaches a pixel after lane `l-1` (the splat
behind it) has already updated that pixel's state. Ported from LichtFeld-Studio's
`blend_backward_cu` (`fastgs/rasterization/include/kernels_backward.cuh`),
Taming-3DGS lineage.

Per-pixel state must live in `@localmem`, which is what forces the second change:
instead of `∇render!`'s per-channel `accum_rec`/`last_color`/`last_α`, each pixel
carries a **single scalar** `W = ∂L/∂T`. The two are algebraically identical —
with `Sⱼ` the code's `accum_rec`,

    W_{j+1} = Sⱼ ⋅ g + (bg ⋅ g)·∏_{m>j}(1 - αₘ),   ∏_{m>j}(1 - αₘ) = T_final / T_{j+1}

so `Tⱼ(cⱼ ⋅ g) - Tⱼ·W_{j+1}` expands term for term into the `vα` built at
`∇render!`'s `accum_rec` loop *plus* its separate background correction. One
scalar therefore replaces all four quantities. Seed: `W = bg ⋅ g`.

Every loop bound below derives from `tile_n`, which is workgroup-uniform, and no
guard wraps a `@synchronize()` — the convergence rule KA requires.
"""
@kernel unsafe_indices=true cpu=false inbounds=true function ∇render_wavefront!(
    # Outputs.
    vcolors::AbstractMatrix{Float32},
    vopacities::AbstractMatrix{Float32},
    vconics::AbstractMatrix{Float32},
    vmeans_2d::AbstractMatrix{Float32},
    # Component-wise `Σₚ |∂L/∂μ|`, accumulated only when given.
    vmeans_2d_abs::A,
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
    grid::SVector{2, Int32}, block::SVector{2, Int32},
    ::Val{tile_size}, ::Val{group_size},
) where {tile_size, group_size, channels, A <: Maybe{AbstractMatrix{Float32}}}
    gid = @index(Group, Linear)  # 1-based tile index ≡ `render!`'s `range_idx`.
    lane = @index(Local, Linear) # 1-based; one splat per lane.

    range = (Int32(ranges[1, gid]), Int32(ranges[2, gid]))
    tile_n = range[2] - range[1]
    # Uniform across the workgroup, so returning here skips no barrier.
    tile_n ≤ 0i32 && return

    # Tile origin, 0-based. Matches `duplicate_with_keys!`'s `tile = y*grid[1] + x`.
    t0 = Int32(gid) - 1i32
    pix_min = SVector{2, Int32}(
        (t0 % grid[1]) * block[1],
        (t0 ÷ grid[1]) * block[2])

    # Per-pixel state, indexed by 1-based tile-local rank (x fastest).
    s_last = @localmem UInt32 tile_size
    s_T = @localmem Float32 tile_size
    s_W = @localmem Float32 tile_size
    s_g = @localmem SVector{channels, Float32} tile_size

    @unroll for k in 0i32:(Int32(tile_size ÷ group_size) - 1i32)
        p = lane + k * group_size
        px = pix_min[1] + (p - 1i32) % block[1] + 1i32
        py = pix_min[2] + (p - 1i32) ÷ block[1] + 1i32
        # Out-of-frame pixels of a partial tile get `s_last = 0`: no rank ever
        # clears it, so they are skipped without a separate guard.
        inside = px ≤ resolution[1] && py ≤ resolution[2]
        g = inside ? vpixels[px, py] : zeros(SVector{channels, Float32})
        s_last[p] = inside ? n_contrib[px, py] : 0u32
        s_T[p] = inside ? accum_α[px, py] : 0f0
        s_g[p] = g
        s_W[p] = bg_color ⋅ g
    end
    @synchronize()

    for batch_base in 0i32:Int32(group_size):(tile_n - 1i32)
        n_batch = min(tile_n - batch_base, Int32(group_size))
        # 0-based index from the front; lane 1 takes the deepest splat left.
        rank = tile_n - batch_base - lane
        valid = lane ≤ n_batch

        gaussian_id = 0u32
        xy = zeros(SVector{2, Float32})
        conic = zeros(SVector{3, Float32})
        color = zeros(SVector{channels, Float32})
        opacity = 0f0
        if valid
            gaussian_id = gaussian_values_sorted[range[1] + rank + 1i32]
            xy = means_2d[gaussian_id]
            opacity = opacities[gaussian_id]
            conic = conics[gaussian_id]
            color = rgb_features[gaussian_id]
        end

        vcolor_acc = zeros(SVector{channels, Float32})
        vconic_acc = zeros(SVector{3, Float32})
        vmean_acc = zeros(SVector{2, Float32})
        vmean_abs_acc = zeros(SVector{2, Float32})
        vopacity_acc = 0f0
        contributed = false

        for diagonal in 0i32:(n_batch + Int32(tile_size) - 2i32)
            p = diagonal - lane + 2i32
            if valid && 1i32 ≤ p ≤ Int32(tile_size) && rank < Int32(s_last[p])
                pix = SVector{2, Int32}(
                    pix_min[1] + (p - 1i32) % block[1],
                    pix_min[2] + (p - 1i32) ÷ block[1])
                δ = xy .- pix
                σ = conic[2] * δ[1] * δ[2] +
                    0.5f0 * (conic[1] * δ[1]^2 + conic[3] * δ[2]^2)
                if σ ≥ 0f0
                    G = exp(-σ)
                    α = min(0.99f0, opacity * G)
                    if α ≥ (1f0 / 255f0)
                        contributed = true
                        # `T` runs backwards: `s_T` holds the transmittance
                        # *after* this splat, so divide to recover it before.
                        T = s_T[p] / (1f0 - α)
                        fac = α * T
                        g = s_g[p]

                        vcolor_acc = vcolor_acc .+ fac .* g
                        cg = color ⋅ g

                        W = s_W[p]
                        vα = T * (cg - W)
                        vσ = -opacity * G * vα

                        vconic_acc = vconic_acc + SVector{3, Float32}(
                            0.5f0 * vσ * δ[1]^2,
                            0.5f0 * vσ * δ[1] * δ[2],
                            0.5f0 * vσ * δ[2]^2)

                        vxy = SVector{2, Float32}(
                            vσ * (conic[1] * δ[1] + conic[2] * δ[2]),
                            vσ * (conic[2] * δ[1] + conic[3] * δ[2]))
                        vmean_acc = vmean_acc + vxy
                        if A !== Nothing
                            # `abs` per pixel, *before* summing - the AbsGS
                            # criterion needs `Σₚ|∂L/∂μ|`, not `|Σₚ ∂L/∂μ|`.
                            vmean_abs_acc = vmean_abs_acc + abs.(vxy)
                        end
                        vopacity_acc += G * vα

                        s_T[p] = T
                        s_W[p] = α * cg + (1f0 - α) * W
                    end
                end
            end
            @synchronize()
        end

        if valid && contributed
            @unroll for c in 1i32:channels
                @atomic vcolors[c, gaussian_id] += vcolor_acc[c]
            end
            @atomic vmeans_2d[1, gaussian_id] += vmean_acc[1]
            @atomic vmeans_2d[2, gaussian_id] += vmean_acc[2]
            if A !== Nothing
                @atomic vmeans_2d_abs[1, gaussian_id] += vmean_abs_acc[1]
                @atomic vmeans_2d_abs[2, gaussian_id] += vmean_abs_acc[2]
            end
            @atomic vconics[1, gaussian_id] += vconic_acc[1]
            @atomic vconics[2, gaussian_id] += vconic_acc[2]
            @atomic vconics[3, gaussian_id] += vconic_acc[3]
            @atomic vopacities[gaussian_id] += vopacity_acc
        end
        # The next batch overwrites `s_T`/`s_W`: no lane may run ahead into it.
        @synchronize()
    end
end

quat_scale_to_cov(q::SVector{4, Float32}, scale::SVector{3, Float32}) =
    quat_scale_to_cov(unnorm_quat2rot(q), scale)

function quat_scale_to_cov(R::SMatrix{3, 3, Float32, 9}, scale::SVector{3, Float32})
    M = R * sdiagm(scale...)
    return M * M'
end

"""
`vR_extra` is an additional cotangent on the rotation matrix, coming from
consumers of `R` other than the covariance (the rendered normal channel uses
one of its columns): it is added to the covariance path's own `vR` so both
share a single quaternion pullback.
"""
function ∇quat_scale_to_cov(
    q::SVector{4, Float32}, scale::SVector{3, Float32},
    R::SMatrix{3, 3, Float32, 9}, vΣ::SMatrix{3, 3, Float32, 9},
    vR_extra::SMatrix{3, 3, Float32, 9} = zeros(SMatrix{3, 3, Float32, 9}),
)
    S = sdiagm(scale...)
    M = R * S

    vM = (vΣ + vΣ') * M
    vR = vM * S + vR_extra

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
    q = q * inv_norm # Unit quaternion: the projection below requires it.
    w, x, y, z = q

    vqn = SVector{4, Float32}(
        2f0 * (
            x * (vR[3, 2] - vR[2, 3]) +
            y * (vR[1, 3] - vR[3, 1]) +
            z * (vR[2, 1] - vR[1, 2])
        ),
        2f0 * (
            -2f0 * x * (vR[2, 2] + vR[3, 3]) +
            y * (vR[2, 1] + vR[1, 2]) +
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

function ∇add_blur(
    compensation::Float32, vcompensation::Float32,
    Σ_2D_blur::SMatrix{2, 2, Float32, 4}, ϵ::Float32,
)
    det_Σ_blur =
        Σ_2D_blur[1, 1] * Σ_2D_blur[2, 2] -
        Σ_2D_blur[1, 2] * Σ_2D_blur[2, 1]
    vsqrt_comp = 0.5f0 * vcompensation / (compensation + 1f-6)
    comp_tmp = 1f0 - compensation^2
    return SMatrix{2, 2, Float32, 4}(
        vsqrt_comp * (comp_tmp * Σ_2D_blur[1, 1] - ϵ * det_Σ_blur),
        vsqrt_comp * comp_tmp * Σ_2D_blur[2, 1],
        vsqrt_comp * comp_tmp * Σ_2D_blur[1, 2],
        vsqrt_comp * (comp_tmp * Σ_2D_blur[2, 2] - ϵ * det_Σ_blur),
    )
end

@inbounds @inline function max_eigval_2D(
    Σ_2D::SMatrix{2, 2, Float32, 4}, det::Float32,
)
    mid = 0.5f0 * (Σ_2D[1, 1] + Σ_2D[2, 2])
    return mid + sqrt(max(0.1f0, mid * mid - det))
end
