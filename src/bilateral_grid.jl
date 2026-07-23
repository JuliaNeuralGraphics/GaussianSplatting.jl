"""
Bilateral grid appearance modeling (as implemented in LichtFeld Studio,
after "Bilateral Guided Radiance Field Processing"): each train image owns a
low-res `(x, y, guidance)` grid of 3×4 affine color transforms, applied to the
render before the photometric loss. Per-view exposure / white-balance drift is
absorbed by the grids instead of being baked into the Gaussians as color-shift
floaters. Training-time only: evaluation & viewing use the raw render.

The grids are trained jointly with the Gaussians (see `step!`): the current
view's grid via [`bilateral_slice`](@ref), all grids via the [`tv_loss`](@ref)
smoothness prior.
"""
struct BilateralGrid{G <: AbstractArray{Float32, 5}, F}
    # (x, y, guidance, 12 affine coefficients, train image).
    # Coefficient `(d - 1) * 4 + c` maps input channel `c` (r, g, b, 1)
    # to output channel `d`.
    grids::G
    optimizer::NU.Adam
    scheduler::F
end

function BilateralGrid(kab, n_images::Int, opt_params)
    gx, gy, gz = opt_params.bilateral_grid_size
    grids = zeros(Float32, gx, gy, gz, 12, n_images)
    for di in 1:3 # Identity transform in every cell.
        grids[:, :, :, (di - 1) * 4 + di, :] .= 1f0
    end
    grids = adapt(kab, grids)
    optimizer = NU.Adam(kab, grids; lr=opt_params.bilateral_grid_lr, ϵ=1f-15)
    scheduler = bilateral_grid_scheduler(
        opt_params.bilateral_grid_lr, opt_params.bilateral_grid_lr_steps)
    return BilateralGrid(grids, optimizer, scheduler)
end

"""
LichtFeld's bilateral grid schedule: linear warmup from 1% of `lr` over the
first 1000 steps (the grids stay near-identity while the Gaussians are still
random), then exponential decay to 1% of `lr` by `steps`.
"""
function bilateral_grid_scheduler(
    lr::Float32, steps::Int;
    warmup_steps::Int = 1000, warmup_start::Float32 = 0.01f0,
    final_factor::Float32 = 0.01f0,
)
    decay = lr_exp_scheduler(lr, final_factor * lr, steps)
    function _scheduler(step::Int)
        warmup = step < warmup_steps ?
            warmup_start + (1f0 - warmup_start) * Float32(step / warmup_steps) :
            1f0
        return warmup * decay(step)
    end
    return _scheduler
end

# Grayscale (ITU-R BT.601) coefficients for the guidance coordinate.
const BGRID_C2G = SVector{3, Float32}(0.299f0, 0.587f0, 0.114f0)

"""
    bilateral_slice(image, grid)

Apply per-pixel affine color transforms trilinearly sampled from `grid` at
`(x, y, grayscale(rgb))` to a `(3, width, height)` `image`.
Differentiable w.r.t. both arguments (including through the guidance
coordinate, except where it saturates).
"""
function bilateral_slice(
    image::AbstractArray{Float32, 3}, grid::AbstractArray{Float32, 4},
)
    out = similar(image)
    _bilateral_slice!(get_backend(image))(
        out, image, grid; ndrange=(size(image, 2), size(image, 3)))
    return out
end

function ChainRulesCore.rrule(::typeof(bilateral_slice),
    image::AbstractArray{Float32, 3}, grid::AbstractArray{Float32, 4},
)
    out = bilateral_slice(image, grid)
    function _bilateral_slice_pullback(Δ)
        kab = get_backend(image)
        ∇image = similar(image)
        ∇grid = KA.zeros(kab, Float32, size(grid))
        _∇bilateral_slice!(kab)(
            ∇image, ∇grid, CRC.unthunk(Δ), image, grid;
            ndrange=(size(image, 2), size(image, 3)))
        return CRC.NoTangent(), ∇image, ∇grid
    end
    return out, _bilateral_slice_pullback
end

"""
Total variation prior over all grids: mean squared difference between
neighboring cells along each grid axis, averaged over axes, coefficients &
images (LichtFeld's normalization). Keeps the affine transforms smooth in
image space & across guidance levels. Zygote-differentiable.
"""
function tv_loss(grids::AbstractArray{Float32, 5})
    gx, gy, gz, _, n = size(grids)
    Δx = @views grids[2:end, :, :, :, :] .- grids[1:(end - 1), :, :, :, :]
    Δy = @views grids[:, 2:end, :, :, :] .- grids[:, 1:(end - 1), :, :, :]
    Δz = @views grids[:, :, 2:end, :, :] .- grids[:, :, 1:(end - 1), :, :]
    return (
        sum(abs2, Δx) / max(1, (gx - 1) * gy * gz) +
        sum(abs2, Δy) / max(1, gx * (gy - 1) * gz) +
        sum(abs2, Δz) / max(1, gx * gy * (gz - 1))) / (12f0 * n)
end

# Interpolation state shared by the forward & backward kernels: sanitized rgb
# & 1-based corner indices with trilinear weights. Coordinates span the full
# grid: pixel (1, 1) hits cell 1, the last pixel hits the last cell; the
# guidance coordinate saturates together with the grayscale clamp.
@inline function _bgrid_coords(image, grid, wi::Int, hi::Int)
    w, h = size(image, 2), size(image, 3)
    gx, gy, gz = size(grid, 1), size(grid, 2), size(grid, 3)

    sr = image[1, wi, hi]
    sg = image[2, wi, hi]
    sb = image[3, wi, hi]
    sr = isfinite(sr) ? sr : 0.5f0
    sg = isfinite(sg) ? sg : 0.5f0
    sb = isfinite(sb) ? sb : 0.5f0

    x = w > 1 ? Float32(wi - 1) / (w - 1) * (gx - 1) : 0f0
    y = h > 1 ? Float32(hi - 1) / (h - 1) * (gy - 1) : 0f0
    guidance = clamp(BGRID_C2G[1] * sr + BGRID_C2G[2] * sg + BGRID_C2G[3] * sb, 0f0, 1f0)
    z = guidance * (gz - 1)

    x0 = gpu_floor(Int, x)
    y0 = gpu_floor(Int, y)
    z0 = clamp(gpu_floor(Int, z), 0, gz - 1)
    x1 = min(x0 + 1, gx - 1)
    y1 = min(y0 + 1, gy - 1)
    z1 = min(z0 + 1, gz - 1)
    fx, fy, fz = x - x0, y - y0, z - z0
    # No guidance gradient where `z` saturates or lands exactly on a cell.
    z_interior = z0 != z && z1 != z
    return (;
        sr, sg, sb, z, z_interior, fx, fy, fz,
        x0=x0 + 1, x1=x1 + 1, y0=y0 + 1, y1=y1 + 1, z0=z0 + 1, z1=z1 + 1)
end

@kernel cpu=false inbounds=true function _bilateral_slice!(
    out::AbstractArray{Float32, 3},
    @Const(image), @Const(grid),
)
    wi, hi = @index(Global, NTuple)
    c = _bgrid_coords(image, grid, wi, hi)

    for di in 1:3
        acc = 0f0
        for si in 1:4
            ci = (di - 1) * 4 + si
            c00 = grid[c.x0, c.y0, c.z0, ci] * (1f0 - c.fx) + grid[c.x1, c.y0, c.z0, ci] * c.fx
            c10 = grid[c.x0, c.y1, c.z0, ci] * (1f0 - c.fx) + grid[c.x1, c.y1, c.z0, ci] * c.fx
            c01 = grid[c.x0, c.y0, c.z1, ci] * (1f0 - c.fx) + grid[c.x1, c.y0, c.z1, ci] * c.fx
            c11 = grid[c.x0, c.y1, c.z1, ci] * (1f0 - c.fx) + grid[c.x1, c.y1, c.z1, ci] * c.fx
            v = (c00 * (1f0 - c.fy) + c10 * c.fy) * (1f0 - c.fz) +
                (c01 * (1f0 - c.fy) + c11 * c.fy) * c.fz
            acc += v * (si == 1 ? c.sr : si == 2 ? c.sg : si == 3 ? c.sb : 1f0)
        end
        out[di, wi, hi] = isfinite(acc) ? acc : 0.5f0
    end
end

@kernel cpu=false inbounds=true function _∇bilateral_slice!(
    # Outputs.
    ∇image::AbstractArray{Float32, 3},
    ∇grid::AbstractArray{Float32, 4},
    # Inputs.
    @Const(Δ), @Const(image), @Const(grid),
)
    wi, hi = @index(Global, NTuple)
    c = _bgrid_coords(image, grid, wi, hi)
    gz = size(grid, 3)

    Δr = Δ[1, wi, hi]
    Δg = Δ[2, wi, hi]
    Δb = Δ[3, wi, hi]
    Δr = isfinite(Δr) ? Δr : 0f0
    Δg = isfinite(Δg) ? Δg : 0f0
    Δb = isfinite(Δb) ? Δb : 0f0

    ∇sr, ∇sg, ∇sb = 0f0, 0f0, 0f0
    ∇z = 0f0 # Gradient w.r.t. the guidance coordinate `z`.
    for corner in 0:7
        xc, yc, zc = corner & 1, (corner >> 1) & 1, (corner >> 2) & 1
        xi = xc == 1 ? c.x1 : c.x0
        yi = yc == 1 ? c.y1 : c.y0
        zi = zc == 1 ? c.z1 : c.z0
        wxy = (xc == 1 ? c.fx : 1f0 - c.fx) * (yc == 1 ? c.fy : 1f0 - c.fy)
        wt = wxy * (zc == 1 ? c.fz : 1f0 - c.fz)
        dwdz = wxy * (zc == 1 ? 1f0 : -1f0) * (gz - 1)

        for di in 1:3
            gout = di == 1 ? Δr : di == 2 ? Δg : Δb
            for si in 1:4
                ci = (di - 1) * 4 + si
                v = grid[xi, yi, zi, ci]
                gb = (si == 1 ? c.sr : si == 2 ? c.sg : si == 3 ? c.sb : 1f0) * gout

                si == 1 && (∇sr += v * wt * gout)
                si == 2 && (∇sg += v * wt * gout)
                si == 3 && (∇sb += v * wt * gout)
                ∇z += dwdz * v * gb
                @atomic ∇grid[xi, yi, zi, ci] += wt * gb
            end
        end
    end

    ∇z *= c.z_interior
    ∇image[1, wi, hi] = ∇sr + BGRID_C2G[1] * ∇z
    ∇image[2, wi, hi] = ∇sg + BGRID_C2G[2] * ∇z
    ∇image[3, wi, hi] = ∇sb + BGRID_C2G[3] * ∇z
end
