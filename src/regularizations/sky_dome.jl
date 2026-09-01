"""
Sky dome: a frozen shell of Gaussians at a large radius, rendered in its own
pass and composited *behind* the scene.

The sky is the one part of an outdoor scene the Gaussians cannot represent
honestly. It has no parallax, so no finite placement is multi-view consistent,
yet the photometric loss still demands that sky pixels reach `α = 1` with
*something*. Without a far-field model the only thing available is a near,
opaque splat — a floater whose sole job is to paint the sky, and which is
invisible from the training views that created it.

The dome gives that content somewhere to go:

- **Frozen geometry.** Positions, scales, rotations & opacities are never
  optimized. Only the colors learn. A dome Gaussian therefore cannot drift
  into the scene and become the very floater it exists to prevent.
- **Shape.** The alpha cotangent the composite feeds back is indiscriminate: it
  removes a floater and a piece of real ground with equal enthusiasm, because a
  free parallax-free background explains *any* not-yet-opaque pixel more
  cheaply than geometry can. A full `:sphere` therefore tends to absorb
  ground — and its colors then learn that ground texture, which is
  self-reinforcing. A `:hemisphere` leaves nothing behind downward-looking
  rays, so that geometry has to become opaque on its own. See
  `sky_dome_shape`.
- **SH degree 0.** One RGB value per Gaussian, no view dependence — a
  view-dependent dome could fake parallax and reintroduce the problem.
  Functionally it is a learned spherical texture that happens to be expressed
  as Gaussians, which is what keeps it exportable (see `export_ply`).
- **Separate pass.** Merging the dome into the main Gaussian set would poison
  the rendered depth channel: a dome Gaussian at `z ≈ 500` leaking through a
  2%-transparent scene pixel shifts that pixel's alpha-normalized depth `D/α`
  by ~10 units and wrecks `ssi_depth_loss` & the normal loss. Rendering it
  separately keeps the depth channel purely about scene geometry.

Because the dome sits far behind everything, compositing it as a background is
*exactly* depth-sorted blending, not an approximation — see [`composite_sky`](@ref).
"""

# Shapes `sky_dome_shape` selects between.
const SKY_DOME_SHAPES = (:hemisphere, :sphere)

"""
Neighbors per dome Gaussian in the lattice graph [`diffuse_sky!`](@ref) walks.
Six is the Fibonacci lattice's own coordination number — its cells are hexagonal —
so this is the full ring around a Gaussian.
"""
const SKY_DOME_NEIGHBORS = 6

"""
Gaussian std as a multiple of the lattice spacing.
"""
const SKY_DOME_OVERLAP = 1f0

"""
How much supervision a dome Gaussian needs before [`diffuse_sky!`](@ref)
trusts its color, as a fraction of what the median supervised Gaussian got.
Relative, because the count grows with the run.
"""
const SKY_SUPERVISED_FRACTION = 0.2f0

"""
Frozen Gaussian shell + the rasterizer & optimizer that drive it.
`gaussians.features_dc` is the only trainable array.

`neighbors` & `hits` exist for [`diffuse_sky!`](@ref): the lattice's k-NN graph
(frozen with the geometry) and how many optimizer steps actually supervised each Gaussian.
"""
struct SkyDome{
    M <: GaussianModel, R <: GaussianRasterizer,
    N <: AbstractMatrix{Int32}, H <: AbstractVector{Float32},
}
    gaussians::M
    optimizer::NU.Adam
    rast::R
    radius::Float32
    neighbors::N
    hits::H
    # Angular spacing of the lattice, which sets how many sweeps
    # `diffuse_sky!` needs to cross the dome.
    spacing::Float32
end

"""
`n` roughly equal-area directions on the unit sphere (Fibonacci lattice).

Returns a `(3, n)` matrix of unit vectors & the mean angular spacing between
neighbors, which sizes the Gaussians: `sqrt(4π / n)` is the side of a spherical
"cell" of area `4π / n`.
"""
function fibonacci_sphere(n::Int)
    dirs = Matrix{Float32}(undef, 3, n)
    golden_angle = Float32(π * (3f0 - sqrt(5f0)))
    for i in 1:n
        # Sample `z` uniformly for equal area; the half-offset keeps both poles
        # off the exact ±1 singularity.
        z = 1f0 - 2f0 * (Float32(i) - 0.5f0) / Float32(n)
        r = sqrt(max(1f0 - z^2, 0f0))
        θ = golden_angle * Float32(i - 1)
        dirs[1, i] = r * cos(θ)
        dirs[2, i] = r * sin(θ)
        dirs[3, i] = z
    end
    return dirs, sqrt(4f0 * Float32(π) / Float32(n))
end

"""
    sky_dome_directions(n, shape, up)

`n` directions for the dome, plus the lattice's angular spacing.

`:sphere` is the whole sky; `:hemisphere` keeps only what is at or above the
horizon, measured against `up`. The lattice is generated at double size and
then cut, so `n` means "Gaussians actually in the dome" either way and the
returned spacing — a property of the lattice, not of the surviving subset —
still sizes them correctly.
"""
function sky_dome_directions(n::Int, shape::Symbol, up::SVector{3, Float32})
    shape in SKY_DOME_SHAPES ||
        error("Invalid sky dome shape: `$shape` ∉ $SKY_DOME_SHAPES.")
    shape ≡ :sphere && return fibonacci_sphere(n)

    dirs, spacing = fibonacci_sphere(2 * n)
    up = normalize(up)
    kept = [i for i in axes(dirs, 2) if
        dirs[1, i] * up[1] + dirs[2, i] * up[2] + dirs[3, i] * up[3] ≥ 0f0]
    return dirs[:, kept], spacing
end

"""
The `(k, n)` `Int32` k-NN graph of the dome lattice.

Built on the unit directions, whose chord distance is monotone in angle, so a
Euclidean `KDTree` returns the angular neighbors. Frozen with the geometry:
the dome is never densified, so this is computed once.
"""
function sky_dome_neighbors(dirs::Matrix{Float32}, k::Int = SKY_DOME_NEIGHBORS)
    n = size(dirs, 2)
    k = clamp(k, 0, n - 1)
    neighbors = Matrix{Int32}(undef, k, n)
    k == 0 && return neighbors

    idxs, _ = knn(KDTree(dirs), dirs, k + 1, true #= sort results =#)
    for i in 1:n
        # The first hit is the query point itself.
        neighbors[:, i] .= Int32.(@view(idxs[i][2:(k + 1)]))
    end
    return neighbors
end

"""
    SkyDome(kab, camera, opt_params; center, radius, color)

Build the dome around `center` at `radius`, with Gaussians sized by
[`SKY_DOME_OVERLAP`](@ref) so the shell is opaque and hole-free.
`up` orients the cut when `sky_dome_shape` is `:hemisphere`.

The dome's rasterizer carries its own far plane (`4 * radius`), since the
default one (1000) would cull the entire shell for any sizeable scene.
"""
function SkyDome(
    kab, camera::Camera, opt_params::OptimizationParams;
    center::SVector{3, Float32},
    radius::Float32,
    up::SVector{3, Float32} = SVector{3, Float32}(0f0, 0f0, 1f0),
    color::SVector{3, Float32} = SVector{3, Float32}(0.5f0, 0.5f0, 0.5f0),
)
    n = opt_params.sky_dome_points
    n > 0 || throw(ArgumentError("`sky_dome_points=$n` must be positive."))

    dirs, spacing = sky_dome_directions(n, opt_params.sky_dome_shape, up)
    n = size(dirs, 2) # The cut lands near, but not exactly on, `n`.

    points = dirs .* radius .+ Array(center)
    colors = repeat(Array(color), 1, n)
    # Isotropic & big enough to overlap its neighbors.
    scales = fill(radius * spacing * SKY_DOME_OVERLAP, 3, n)

    gaussians = GaussianModel(kab, points, colors, log.(scales); max_sh_degree=0)
    # Opaque: the render kernel caps `α` at 0.99 regardless, and a transparent
    # dome would just let the (black) background through as holes in the sky.
    gaussians.opacities .= inverse_sigmoid(0.99f0)

    optimizer = NU.Adam(kab, gaussians.features_dc; lr=opt_params.sky_dome_lr, ϵ=1f-15)
    rast = GaussianRasterizer(kab, camera; mode=:rgb, far_plane=4f0 * radius)
    neighbors = adapt(kab, sky_dome_neighbors(dirs))
    hits = KA.zeros(kab, Float32, n)
    return SkyDome(gaussians, optimizer, rast, radius, neighbors, hits, spacing)
end

"""
Dome radius for a dataset: `sky_dome_radius_factor` camera extents away, so it
scales with the scene rather than with world units.

Clamped to keep the whole shell inside the *scene* rasterizer's far plane too,
which matters for anything that later renders the merged set (`export_ply`).
"""
function sky_dome_radius(rast::GaussianRasterizer, opt_params::OptimizationParams, extent::Float32)
    radius = opt_params.sky_dome_radius_factor * extent
    # Leave room for the camera's own offset from the dome center.
    return min(radius, 0.8f0 * rast.far_plane - extent)
end

memory_usage(sky::SkyDome) =
    memory_usage(sky.gaussians) +
    memory_usage(sky.optimizer) +
    memory_usage(sky.rast) +
    memory_usage(sky.neighbors) +
    memory_usage(sky.hits)

function KA.unsafe_free!(sky::SkyDome)
    KA.unsafe_free!(sky.gaussians)
    free_optimizer!(sky.optimizer)
    KA.unsafe_free!(sky.rast)
    KA.unsafe_free!(sky.neighbors)
    KA.unsafe_free!(sky.hits)
    return
end

Base.length(sky::SkyDome) = length(sky.gaussians)

"""
    record_sky_supervision!(sky, ∇features_dc)

Count the dome Gaussians this step's gradient actually reached.

The dome's gradient is dense & exactly zero wherever no pixel of the view was
explained by that Gaussian, so a nonzero entry means "some train view saw it,
through the scene" — occlusion included, which is the property
[`diffuse_sky!`](@ref) needs and a frustum test would not give.
"""
function record_sky_supervision!(sky::SkyDome, ∇features_dc::AbstractArray{Float32, 3})
    isempty(sky.hits) && return
    _accumulate_sky_hits!(get_backend(sky.gaussians))(
        sky.hits, ∇features_dc; ndrange=length(sky.hits))
    return
end

@kernel cpu=false inbounds=true function _accumulate_sky_hits!(
    hits::AbstractVector{Float32}, @Const(∇features_dc),
)
    i = @index(Global)
    ∇ =
        abs(∇features_dc[1, 1, i]) +
        abs(∇features_dc[2, 1, i]) +
        abs(∇features_dc[3, 1, i])
    ∇ > 0f0 && (hits[i] += 1f0)
end

"""
    diffuse_sky!(sky)

Extrapolate the learned sky over the part of the dome no train view covers.

A dataset that does not photograph the zenith still renders it: those Gaussians
keep `sky_init_color` while their supervised neighbors move to the real sky, and
the boundary between the two is a visible seam in any view that looks up.

The fix is an inpaint over the lattice graph. Well supervised Gaussians
([`SKY_SUPERVISED_FRACTION`](@ref)) are Dirichlet boundary values — never written —
and every other one is a weighted Jacobi average of its neighbors.
Confidence starts at 1 on the boundary and 0 elsewhere, so the early sweeps
advance a front one ring per sweep (the fill) and the later ones relax what is
behind it (the smoothing): the result is the discrete harmonic extension of the
observed sky, which meets the boundary with no step in value.
"""
function diffuse_sky!(sky::SkyDome; smooth_sweeps::Int = 32)
    n = length(sky)
    (n == 0 || size(sky.neighbors, 1) == 0) && return

    kab = get_backend(sky.gaussians)
    # Nothing has been seen yet on an untrained dome: there is no sky to
    # extrapolate from, so leave it alone.
    # The median is over the supervised Gaussians only.
    seen = filter(>(0f0), adapt(CPU(), sky.hits))
    isempty(seen) && return
    min_hits = SKY_SUPERVISED_FRACTION * median(seen)

    # A front advances one ring per sweep, so crossing the widest possible hole
    # — half the dome — takes `π / spacing` of them.
    sweeps = ceil(Int, Float32(π) / max(sky.spacing, 1f-6)) + smooth_sweeps
    iseven(sweeps) || (sweeps += 1) # Land the result back in `features_dc`.

    features = sky.gaussians.features_dc
    src, dst = features, similar(features)
    w_src = KA.zeros(kab, Float32, n)
    w_dst = KA.zeros(kab, Float32, n)

    diffuse! = _diffuse_sky!(kab)
    for _ in 1:sweeps
        diffuse!(dst, w_dst, src, w_src, sky.neighbors, sky.hits, min_hits; ndrange=n)
        src, dst = dst, src
        w_src, w_dst = w_dst, w_src
    end
    @assert src ≡ features # Even sweep count.

    KA.unsafe_free!(dst)
    KA.unsafe_free!(w_src)
    KA.unsafe_free!(w_dst)
    return
end

@kernel cpu=false inbounds=true function _diffuse_sky!(
    # Outputs.
    features_dst::AbstractArray{Float32, 3},
    w_dst::AbstractVector{Float32},
    # Inputs.
    @Const(features_src), @Const(w_src), @Const(neighbors), @Const(hits),
    min_hits::Float32,
)
    i = @index(Global)

    if hits[i] ≥ min_hits
        # Supervised: a boundary value, carried through untouched.
        @unroll for c in 1:3
            features_dst[c, 1, i] = features_src[c, 1, i]
        end
        w_dst[i] = 1f0
    else
        k = size(neighbors, 1)
        acc = zero(SVector{3, Float32})
        Σw = 0f0
        for ki in 1:k
            j = neighbors[ki, i]
            w = w_src[j]
            w > 0f0 || continue # Nothing has reached this neighbor yet.
            acc += w * SVector{3, Float32}(features_src[1, 1, j], features_src[2, 1, j], features_src[3, 1, j])
            Σw += w
        end

        if Σw > 0f0
            acc = acc ./ Σw
            @unroll for c in 1:3
                features_dst[c, 1, i] = acc[c]
            end
            # Confidence of an average is the confidence of what it averaged:
            # full only where the whole ring is, which is what keeps the front
            # ahead of the values it is still relaxing.
            w_dst[i] = Σw / Float32(k)
        else
            @unroll for c in 1:3
                features_dst[c, 1, i] = features_src[c, 1, i]
            end
            w_dst[i] = w_src[i]
        end
    end
end

"""
    render_sky(sky, camera; rast)

Render the dome's RGB for `camera`. `features_dc` is passed through so the
result is differentiable w.r.t. the dome colors; every other dome array is a
frozen constant and is deliberately left out of the AD graph.

`rast` defaults to the dome's own rasterizer, which is sized to the dataset.
Views at another resolution (the GUI) pass their own so the training one is not
rebuilt back and forth.
"""
function render_sky(
    sky::SkyDome, camera::Camera, features_dc::AbstractArray{Float32, 3};
    rast::GaussianRasterizer = sky.rast,
)
    gs = sky.gaussians
    return rast(
        gs.points, gs.opacities, gs.scales, gs.rotations,
        features_dc, gs.features_rest;
        camera, sh_degree=0)
end

render_sky(sky::SkyDome, camera::Camera; kwargs...) =
    render_sky(sky, camera, sky.gaussians.features_dc; kwargs...)

"""
    sky_view_rasterizer(kab, sky, camera)

An `:rgb` rasterizer for the dome at `camera`'s resolution, carrying the dome's
far plane. For render paths outside training.
"""
sky_view_rasterizer(kab, sky::SkyDome, camera::Camera) =
    GaussianRasterizer(kab, camera; mode=:rgb, far_plane=4f0 * sky.radius)

"""
    composite_sky!(rast, sky, camera; sky_rast)

In-place [`composite_sky`](@ref) for the non-AD render paths: adds the dome
into rows 1:3 of `rast.image`, which is what `gl_texture`/`to_image` read.
Requires a `:rgbd`/`:rgbdn` `rast` for the alpha row; the depth row is
untouched, so depth display & picking are unaffected.
"""
function composite_sky!(
    rast::GaussianRasterizer, sky::SkyDome, camera::Camera;
    sky_rast::GaussianRasterizer = sky.rast,
)
    rast.mode == :rgb && return rast.image

    sky_rgb = render_sky(sky, camera; rast=sky_rast)
    alpha = @view rast.image[5, :, :]
    @view(rast.image[1:3, :, :]) .+=
        reshape(1f0 .- alpha, 1, size(alpha)...) .* sky_rgb
    return rast.image
end

"""
    composite_sky(image, alpha, sky_rgb)

Composite the dome behind a scene render: `image + (1 - α)·sky`.

`image` is the scene's `Σ cᵢ·αᵢ·Tᵢ` (rendered over a zero background) and
`alpha` is channel 5, which blends a constant-1 feature and therefore equals
`1 - T_final` exactly. The result is `Σ cᵢ·αᵢ·Tᵢ + T_final·sky`, i.e. ordinary
back-to-front blending with the dome last — exact rather than approximate,
because the dome really is behind everything.

The `(1 - α)` factor also routes an alpha cotangent from the photometric loss
back into the scene rasterizer: wherever the dome explains a pixel better than
a splat does, L1/SSIM push that splat's alpha down on their own. That is what
makes the dome remove floaters instead of merely tolerating them.
"""
composite_sky(
    image::AbstractArray{Float32, 3},
    alpha::AbstractMatrix{Float32},
    sky_rgb::AbstractArray{Float32, 3},
) = image .+ reshape(1f0 .- alpha, 1, size(alpha)...) .* sky_rgb

"""
Merge the dome into `gs` for export: one Gaussian set that any external viewer
renders correctly, since the dome sorts behind the scene by construction.

The dome's SH is degree 0, so its higher bands are zero-padded up to the
scene's `max_sh_degree` — a constant color under any viewer's SH evaluation.
"""
merge_sky(gs::GaussianModel, sky::SkyDome) = merge_sky(gs, sky.gaussians)

function merge_sky(gs::GaussianModel, sky_gs::GaussianModel)
    kab = get_backend(gs)
    n = length(sky_gs)

    features_rest = KA.zeros(kab, Float32, (3, size(gs.features_rest, 2), n))
    scales = size(gs.scales, 1) == 1 ? # Match the scene's isotropic layout.
        mean(sky_gs.scales; dims=1) : sky_gs.scales

    return GaussianModel(
        hcat(gs.points, sky_gs.points),
        cat(gs.features_dc, sky_gs.features_dc; dims=3),
        cat(gs.features_rest, features_rest; dims=3),
        hcat(gs.scales, scales),
        hcat(gs.rotations, sky_gs.rotations),
        hcat(gs.opacities, sky_gs.opacities),
        nothing,
        gs.sh_degree, gs.max_sh_degree)
end

"""
    zero_alpha_loss(alpha, weight)

Pull the scene's accumulated alpha to zero where `weight` says nothing should be:
sky rays (`sky/`) and everything outside a coverage mask (`masks/`, see `masking.jl`).

`alpha` must be the raw channel-5 render, not clamped.
"""
function zero_alpha_loss(alpha::AbstractMatrix{Float32}, weight::AbstractMatrix{Float32})
    Σw = ignore_derivatives(max(sum(weight), 1f0))
    return sum(weight .* alpha .^ 2) / Σw
end

function write_state!(tensors, meta, prefix::String, sky::SkyDome)
    write_state!(tensors, meta, "$prefix.gaussians", sky.gaussians)
    write_state!(tensors, meta, "$prefix.optimizer", sky.optimizer)
    write_scalar!(meta, "$prefix.radius", sky.radius)
    tensors["$prefix.hits"] = adapt(CPU(), sky.hits)
    return
end

function read_state!(sky::SkyDome, ckpt::Checkpoint, prefix::String)
    read_state!(sky.gaussians, ckpt, "$prefix.gaussians")
    read_state!(sky.optimizer, ckpt, "$prefix.optimizer")
    # The lattice is rebuilt from the same params, so `hits` lines up.
    key = "$prefix.hits"
    hits = haskey(ckpt, key) ? tensor(ckpt, key)::Vector{Float32} : Float32[]
    if length(hits) == length(sky.hits)
        copyto!(sky.hits, hits)
    else
        fill!(sky.hits, 1f0)
    end
    return
end
