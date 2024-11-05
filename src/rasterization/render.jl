function render(
    means_2d::AbstractMatrix{Float32},
    conics::AbstractMatrix{Float32},
    opacities::AbstractMatrix{Float32},
    colors::AbstractMatrix{Float32};
    rast::GaussianRasterizer, camera::Camera, background::SVector{3, Float32},
    # Used only for sorting.
    depths::AbstractVector{Float32},
)
    kab = get_backend(rast)
    n = size(means_2d, 2)

    (; width, height) = resolution(camera)
    @assert width % 16 == 0 && height % 16 == 0

    channels = size(colors, 1)
    image = KA.zeros(kab, Float32, (channels, width, height))

    count_tiles_per_gaussian!(kab, Int(BLOCK_SIZE))(
        # Output.
        rast.gstate.tiles_touched,
        # Input.
        _as_T(SVector{2, Float32}, means_2d),
        rast.gstate.radii, rast.grid, BLOCK; ndrange=n)

    cumsum!(
        @view(rast.gstate.points_offset[1:n]),
        @view(rast.gstate.tiles_touched[1:n]))
    # Get total number of tiles touched.
    n_rendered = Int(@allowscalar rast.gstate.points_offset[n])
    n_rendered == 0 && return image

    if length(rast.bstate) < n_rendered
        KA.unsafe_free!(rast.bstate)
        rast.bstate = BinningState(kab, n_rendered)
    end

    # For each instance to be rendered, produce [tile | depth] key
    # and corresponding duplicated Gaussian indices to be sorted.
    duplicate_with_keys!(kab, Int(BLOCK_SIZE))(
        # Output.
        rast.bstate.gaussian_keys_unsorted,
        rast.bstate.gaussian_values_unsorted,
        # Input.
        _as_T(SVector{2, Float32}, means_2d),
        depths,
        rast.gstate.points_offset,
        rast.gstate.radii, rast.grid, BLOCK; ndrange=n)

    sortperm!(
        @view(rast.bstate.permutation[1:n_rendered]),
        @view(rast.bstate.gaussian_keys_unsorted[1:n_rendered]))
    _permute!(kab)(
        rast.bstate.gaussian_keys_sorted, rast.bstate.gaussian_keys_unsorted,
        rast.bstate.permutation; ndrange=n_rendered)
    _permute!(kab)(
        rast.bstate.gaussian_values_sorted, rast.bstate.gaussian_values_unsorted,
        rast.bstate.permutation; ndrange=n_rendered)

    # Identify start-end of per-tile workloads in sorted keys.
    fill!(rast.istate.ranges, 0u32)
    identify_tile_range!(kab, Int(BLOCK_SIZE))(
        rast.istate.ranges, rast.bstate.gaussian_keys_sorted;
        ndrange=n_rendered)

    if channels > 3
        background_tmp = zeros(MVector{channels, Float32})
        background_tmp[1:3] = background
        background = SVector{channels, Float32}(background_tmp)
    end
    render!(kab, (Int.(BLOCK)...,), (width, height))(
        # Outputs.
        image, rast.istate.n_contrib, rast.istate.accum_α,
        # Inputs.
        rast.bstate.gaussian_values_sorted,
        _as_T(SVector{2, Float32}, means_2d),
        opacities,
        _as_T(SVector{3, Float32}, conics),
        _as_T(SVector{channels, Float32}, colors),
        rast.istate.ranges,
        SVector{2, Int32}(width, height),
        background, BLOCK, Val(BLOCK_SIZE))
    return image
end

function ∇render(
    vpixels::AbstractArray{Float32, 3},
    means_2d::AbstractMatrix{Float32},
    conics::AbstractMatrix{Float32},
    opacities::AbstractMatrix{Float32},
    colors::AbstractMatrix{Float32};
    rast::GaussianRasterizer, camera::Camera, background::SVector{3, Float32},
)
    kab = get_backend(rast)
    n = size(means_2d, 2)

    (; width, height) = resolution(camera)
    @assert width % 16 == 0 && height % 16 == 0

    fill!(reinterpret(Float32, rast.gstate.∇means_2d), 0f0)

    vmeans_2d = KA.zeros(kab, Float32, size(means_2d))
    vconics = KA.zeros(kab, Float32, size(conics))
    vopacities = KA.zeros(kab, Float32, size(opacities))
    vcolors = KA.zeros(kab, Float32, size(colors))

    channels = size(colors, 1)
    if channels > 3
        background_tmp = zeros(MVector{channels, Float32})
        background_tmp[1:3] = background
        background = SVector{channels, Float32}(background_tmp)
    end
    ∇render!(kab, (Int.(BLOCK)...,), (width, height))(
        # Outputs.
        vcolors, vopacities, vconics, vmeans_2d,
        # Inputs.
        reshape(reinterpret(SVector{channels, Float32}, vpixels), size(vpixels)[2:3]),
        rast.istate.n_contrib,
        rast.istate.accum_α,
        rast.bstate.gaussian_values_sorted,
        _as_T(SVector{2, Float32}, means_2d),
        opacities,
        _as_T(SVector{3, Float32}, conics),
        _as_T(SVector{channels, Float32}, colors),
        rast.istate.ranges,
        SVector{2, Int32}(width, height), background,
        rast.grid, BLOCK, Val(BLOCK_SIZE))

    # Accumulate for densificaton.
    @view(rast.gstate.∇means_2d[1:n]) .+= _as_T(SVector{2, Float32}, vmeans_2d)
    return vmeans_2d, vconics, vopacities, vcolors
end

function ChainRulesCore.rrule(::typeof(render),
    means_2d::AbstractMatrix{Float32},
    conics::AbstractMatrix{Float32},
    opacities::AbstractMatrix{Float32},
    colors::AbstractMatrix{Float32};
    rast::GaussianRasterizer, camera::Camera, background::SVector{3, Float32},
    depths::AbstractVector{Float32},
)
    image = render(
        means_2d, conics, opacities, colors;
        rast, camera, background, depths)
    function _render_pullback(vpixels)
        ∇ = ∇render(
            vpixels, means_2d, conics, opacities, colors;
            rast, camera, background)
        return (NoTangent, ∇...)
    end
    return image, _render_pullback
end

@kernel cpu=false inbounds=true function render!(
    # Output.
    out_color::AbstractArray{Float32, 3},
    n_contrib::AbstractMatrix{UInt32},
    accum_α::AbstractMatrix{Float32},
    # Input.
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
    collected_conics = @localmem SVector{3, Float32} block_size
    collected_xy = @localmem SVector{2, Float32} block_size
    collected_opacity = @localmem Float32 block_size
    collected_id = @localmem UInt32 block_size

    T = 1f0
    contributor = 0u32
    last_contributor = 0u32

    color = zeros(MVector{channels, Float32})
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
            vα *= T # TODO mull by vaccum_α when supporting :rgbed mode
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
