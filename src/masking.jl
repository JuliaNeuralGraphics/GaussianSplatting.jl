"""
Coverage masks: an optional `masks/` directory next to the dataset images, one
per view. White keeps a pixel, black drops it.

The mask is applied to the *target*, so training sees what it would see if the
images had been masked on disk: the subject where the mask keeps it, the render
pass' own background everywhere else. Outside the mask is therefore supervised,
not dropped from the loss — a region that is merely unscored keeps whatever the
point cloud put there and fades to a haze after the first opacity reset.

Which background that is decides what the photometric term can say. Over black
it says almost nothing: black is what an empty ray and an opaque black gaussian
both render, so the emptied region fills with black floaters that show up from
every other view, and on the subject itself `α` and color are interchangeable
(`c·α + bkgd·T` with `bkgd = 0` is blind to `T`), which leaves the walls
half-transparent. `zero_alpha_loss` on the mask complement
(`mask_opacity_weight`) is what breaks the first tie, and it is all there is.

`random_background` (see `OptimizationParams`) breaks both at full photometric
weight, and [`composite_mask`](@ref) is what makes it correct here: with the
target composited over the *same* color the splats were composited over, a
background that changes every step can only be matched by `α = 0` outside the
mask and `α = 1` inside it.

The terms that read geometry rather than color — depth priors, depth-normal
consistency — are gated by the thresholded mask instead: there is no surface
behind a mask to describe.
"""

"""
Load a mask as a `(width, height)` Float32 weight map in `[0, 1]`, resized to
the training resolution. Mirrors `load_depth_prior`'s layout conventions.

Kept soft rather than thresholded: a resized mask has genuinely fractional
border pixels. Consumers that cannot act on a fraction of a pixel threshold it
themselves (see [`mask_hard`](@ref)). Also loads the sky masks of `sky/`.
"""
function load_mask(path::String, width::Int, height::Int)
    raw = load(path)
    mask = Float32.(Gray.(raw))
    mask = imresize(mask, (height, width))
    return clamp!(permutedims(mask, (2, 1)), 0f0, 1f0)
end

"""
The mask of `image_name` under `masks_dir`, or `nothing` when that view has
none — a partially masked dataset is fine.
"""
function load_view_mask(masks_dir::String, image_name::String, width::Int, height::Int)
    path = joinpath(masks_dir, "$(splitext(image_name)[1]).png")
    isfile(path) || return nothing
    return load_mask(path, width, height)
end

"""
Whether a mask keeps less than a single pixel's worth of weight. Such a view
has no subject, only a uniformly black target, so `ColmapDataset` drops it.
"""
mask_is_empty(mask::AbstractMatrix{Float32}) = sum(mask) < 1f0

# Threshold for uses that cannot act on a fraction of a pixel.
mask_hard(mask::AbstractMatrix{Float32}) = mask .> 0.5f0

# A `(width, height)` mask in the `(width, height, 1, 1)` layout of the images
# the loss & metrics are computed on, so it broadcasts over the channels.
image_mask(mask::AbstractMatrix{Float32}) = reshape(mask, size(mask)..., 1, 1)

"""
A view's target with its mask applied: the image where the mask keeps it, black
outside. What the metrics are computed against — evaluation always renders over
black, so there is no other background to composite over.
"""
apply_mask(image::AbstractArray{Float32, 4}, mask::AbstractMatrix{Float32}) =
    image .* image_mask(mask)

"""
A view's target composited over `background`, the color the render pass put
behind the splats: the image where the mask keeps it, `background` outside.
What the training loss is computed against.

Over black this is exactly [`apply_mask`](@ref). Over the random background of
`random_background` it is what makes the mask a claim about *occupancy* rather
than about color, at the photometric term's own weight: outside the mask the
target no longer depends on what the scene paints there, so only `α = 0`
matches it, and inside the mask the target holds still while the background
moves, so only `α = 1` does.
"""
function composite_mask(
    image::AbstractArray{Float32, 4}, mask::AbstractMatrix{Float32},
    background::SVector{3, Float32},
)
    iszero(background) && return apply_mask(image, mask)
    m = image_mask(mask)
    # Channels are the third axis here, as in `image_mask`'s layout.
    bg = adapt(get_backend(image), reshape(collect(background), 1, 1, 3, 1))
    return image .* m .+ bg .* (1f0 .- m)
end

# Below this a carve is more likely a mask/pose mismatch than a small subject,
# and `ColmapDataset` keeps the cloud whole rather than start from nothing.
const MIN_CARVED_POINTS = 100

"""
Which init points the coverage masks place on the subject: a point that
projects outside the mask in more than `tolerance` of the views that see it is
not part of what the masks keep.

A one-time visual-hull carve of the COLMAP cloud. The loss can only empty the
region the masks exclude one layer at a time — `zero_alpha_loss` reads the
*accumulated* alpha, so whatever hides behind the first opaque surface gets no
gradient at all — whereas a point that is never seeded never has to be removed.
Densification then spends its budget on the subject instead of on the room
around it.

Points that no masked view sees are dropped too: the subject is, by
construction, in front of the cameras.
"""
function carve_points(
    points::AbstractMatrix{Float32}, cameras::Vector{Camera},
    masks::Vector{Maybe{Matrix{Float32}}}; tolerance::Float32 = 0.1f0,
)
    length(cameras) == length(masks) || throw(ArgumentError(
        "`cameras` ($(length(cameras))) & `masks` ($(length(masks))) " *
        "must be the same per-view vectors."))

    # Only masked views carve: an unmasked one is no evidence either way, the
    # same tolerance for a partially masked dataset as everywhere else here.
    views = [
        (c.w2c, c.intrinsics, m)
        for (c, m) in zip(cameras, masks) if m ≢ nothing]

    # `Vector{Bool}`, not a `BitVector`: neighbouring bits of the latter share a
    # word, so the threaded writes below would race on it.
    keep = fill(false, size(points, 2))
    Threads.@threads for i in axes(points, 2)
        p = SVector{3, Float32}(points[1, i], points[2, i], points[3, i])
        seen, outside = 0, 0
        for (w2c, intrinsics, mask) in views
            # `R * p + t`, the rasterizer's `pos_world_to_cam` (`projection.jl`).
            z = w2c[3, 1] * p[1] + w2c[3, 2] * p[2] + w2c[3, 3] * p[3] + w2c[3, 4]
            z > 0f0 || continue
            x = w2c[1, 1] * p[1] + w2c[1, 2] * p[2] + w2c[1, 3] * p[3] + w2c[1, 4]
            y = w2c[2, 1] * p[1] + w2c[2, 2] * p[2] + w2c[2, 3] * p[3] + w2c[2, 4]

            resolution = intrinsics.resolution
            focal, principal = intrinsics.focal, intrinsics.principal
            # Continuous pixel coordinates in `[0, resolution)`, as
            # `perspective_projection` produces them.
            u = focal[1] * x / z + principal[1] * resolution[1]
            v = focal[2] * y / z + principal[2] * resolution[2]
            # Bounds-checked before truncating: `u`/`v` are unbounded (and a
            # degenerate pose can make them `NaN`), which `trunc(Int, _)` is not.
            (0f0 ≤ u < Float32(size(mask, 1)) && 0f0 ≤ v < Float32(size(mask, 2))) ||
                continue
            px, py = trunc(Int, u) + 1, trunc(Int, v) + 1

            seen += 1
            mask[px, py] > 0.5f0 || (outside += 1)
        end
        keep[i] = seen > 0 && outside ≤ tolerance * seen
    end
    return keep
end
