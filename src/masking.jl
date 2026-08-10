"""
Coverage masks: an optional `masks/` directory next to the dataset images, one
per view. White keeps a pixel, black drops it.

The mask is applied to the *target*, so training sees what it would see if the
images had been masked on disk: the subject where the mask keeps it, black
everywhere else. Outside the mask is therefore supervised, not dropped from the
loss — a region that is merely unscored keeps whatever the point cloud put
there and fades to a haze after the first opacity reset.

Color is only half of it. Black is what an empty ray and an opaque black
gaussian both render, so the photometric term alone fills the emptied region
with black floaters that show up from every other view. `zero_alpha_loss` on
the mask complement (`mask_opacity_weight`) is what says *empty* rather than
*dark*, exactly as it does for sky masks.

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
outside. What both the loss and the metrics are computed against.
"""
apply_mask(image::AbstractArray{Float32, 4}, mask::AbstractMatrix{Float32}) =
    image .* image_mask(mask)
