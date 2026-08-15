"""
Coverage masks: an optional `masks/` directory next to the dataset images, one per view.
White keeps a pixel, black drops it.

The mask is applied to the *target*, so training sees what it would see if the
images had been masked on disk:
the subject where the mask keeps it, the render pass' own background everywhere else.

Outside the mask is therefore supervised, not dropped from the loss —
a region that is merely unscored keeps whatever the
point cloud put there and fades to a haze after the first opacity reset.

The terms that read geometry rather than color — depth priors, depth-normal
consistency — are gated by the thresholded mask instead: there is no surface
behind a mask to describe.
"""

# Below this a carve is more likely a mask/pose mismatch than a small subject,
# and `ColmapDataset` keeps the cloud whole rather than start from nothing.
const MIN_CARVED_POINTS = 100

"""
Load a mask as a `(width, height)` Float32 weight map in `[0, 1]`, at the
training resolution **or smaller**. Mirrors `load_depth_prior`'s conventions.

If mask is smaller, the expansion is left to [`device_map`](@ref) — done on the device it costs
neither the host allocation nor the bus traffic a full-resolution copy would.
Only a mask stored at or above the training resolution is resampled here,
where `imresize` antialiases the downscale.

Kept soft rather than thresholded: a resampled mask has fractional border pixels.
Consumers that cannot act on a fraction of a pixel threshold it
themselves (see [`mask_hard`](@ref)). Also loads the sky masks of `sky/`.
"""
function load_mask(path::String, width::Int, height::Int)
    mask = load_mask_raw(path)
    all(size(mask) .≤ (width, height)) && return mask
    return clamp!(fit_resolution(mask, (width, height)), 0f0, 1f0)
end

"""
A mask at the resolution it is stored at.
Original mask resolution is used for emptiness check and [`carve_points`](@ref)).
"""
load_mask_raw(path::String) = clamp!(permutedims(Float32.(Gray.(load(path))), (2, 1)), 0f0, 1f0)

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
A view's target with its mask applied: the image where the mask keeps it, black outside.
"""
apply_mask(image::AbstractArray{Float32, 4}, mask::AbstractMatrix{Float32}) =
    image .* image_mask(mask)

"""
A view's target composited over `background`, the color the render pass put
behind the splats: the image where the mask keeps it, `background` outside.
What the training loss is computed against.
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

"""
Remove points from initial point-cloud that are not covered by the masks stored at `mask_paths`.
A point that projects outside the mask in more than `tolerance` of the views that see it is
not part of what the masks keep.
"""
function carve_points(
    points::AbstractMatrix{Float32}, cameras::Vector{Camera},
    mask_paths::Vector{Maybe{String}}; tolerance::Float32 = 0.1f0,
)
    length(cameras) == length(mask_paths) || throw(ArgumentError(
        "`cameras` ($(length(cameras))) & `mask_paths` ($(length(mask_paths))) " *
        "must be the same per-view vectors."))

    n_points = size(points, 2)
    seen = zeros(Int, n_points)
    outside = zeros(Int, n_points)

    for (camera, path) in zip(cameras, mask_paths)
        path ≡ nothing && continue
        mask = load_mask_raw(path)
        mask_width, mask_height = size(mask)

        w2c = camera.w2c
        intrinsics = camera.intrinsics
        resolution = intrinsics.resolution
        focal, principal = intrinsics.focal, intrinsics.principal
        # Pixels of the training resolution to pixels of the mask:
        # the mask covers the same frame, at whatever size it is stored.
        sx = Float32(mask_width) / Float32(resolution[1])
        sy = Float32(mask_height) / Float32(resolution[2])

        # `seen`/`outside` are plain `Vector{Int}` & every iteration writes its
        # own index, so the threaded accumulation below cannot race.
        Threads.@threads for i in 1:n_points
            p = SVector{3, Float32}(points[1, i], points[2, i], points[3, i])
            # `R * p + t`, the rasterizer's `pos_world_to_cam` (`projection.jl`).
            z = w2c[3, 1] * p[1] + w2c[3, 2] * p[2] + w2c[3, 3] * p[3] + w2c[3, 4]
            z > 0f0 || continue
            x = w2c[1, 1] * p[1] + w2c[1, 2] * p[2] + w2c[1, 3] * p[3] + w2c[1, 4]
            y = w2c[2, 1] * p[1] + w2c[2, 2] * p[2] + w2c[2, 3] * p[3] + w2c[2, 4]

            # Continuous pixel coordinates in `[0, resolution)`, as
            # `perspective_projection` produces them.
            u = focal[1] * x / z + principal[1] * resolution[1]
            v = focal[2] * y / z + principal[2] * resolution[2]
            # Bounds-checked before truncating: `u`/`v` are unbounded (and a
            # degenerate pose can make them `NaN`), which `trunc(Int, _)` is not.
            (0f0 ≤ u < Float32(resolution[1]) && 0f0 ≤ v < Float32(resolution[2])) || continue
            # `min` guards the last pixel: `u` just under `resolution[1]` can
            # still scale onto `mask_width` itself.
            px = min(trunc(Int, u * sx) + 1, mask_width)
            py = min(trunc(Int, v * sy) + 1, mask_height)

            seen[i] += 1
            mask[px, py] > 0.5f0 || (outside[i] += 1)
        end
    end

    keep = fill(false, n_points)
    Threads.@threads for i in 1:n_points
        keep[i] = seen[i] > 0 && outside[i] ≤ tolerance * seen[i]
    end
    return keep
end
