# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

# Width the view thumbnails are downscaled to (see `with_thumbnails`).
const THUMBNAIL_WIDTH = 128

# How many steps ahead [`ViewLoader`](@ref) reads.
const VIEW_LOOKAHEAD = 3

struct ColmapDataset
    points::Matrix{Float32}
    colors::Matrix{Float32}
    scales::Matrix{Float32}

    # Directory the images are read from (already `_$scale`-suffixed) &
    # the resolution every view is loaded at.
    images_dir::String
    resolution::SVector{2, UInt32}

    train_image_filenames::Vector{String}
    train_cameras::Vector{Camera}
    # Downscaled train images for display (the GUI maps them onto the camera frustums).
    # Empty unless the dataset was loaded `with_thumbnails`.
    # Each is `(channels, width, height)`, like the images `load_image` returns.
    train_thumbnails::Vector{Array{UInt8, 3}}

    # Monocular depth priors for depth supervision.
    train_depth_paths::Vector{Maybe{String}}
    has_depth_priors::Bool
    depths_dir::Maybe{String}

    # Sky masks for sky supervision.
    # Soft weights in `[0, 1]`, so antialiased mask borders contribute
    # proportionally instead of snapping (see `load_mask`).
    train_sky_paths::Vector{Maybe{String}}
    has_sky_masks::Bool
    sky_dir::Maybe{String}

    # Coverage masks, applied to the targets of both splits
    # so training & scoring see the same image (`masking.jl`).
    train_mask_paths::Vector{Maybe{String}}
    test_mask_paths::Vector{Maybe{String}}
    has_masks::Bool
    masks_dir::Maybe{String}

    test_image_filenames::Vector{String}
    test_cameras::Vector{Camera}

    camera_extent::Float32
end

function ColmapDataset(dataset_dir::String;
    scale::Int = 1, holdout::Int = 8, max_extent::Float32 = Inf32,
    with_thumbnails::Bool = false, use_masks::Bool = true,
    carve_with_masks::Bool = true, carve_tolerance::Float32 = 0.1f0,
)
    cameras_file = joinpath(dataset_dir, "sparse", "0", "cameras.bin")
    images_file = joinpath(dataset_dir, "sparse", "0", "images.bin")
    points_file = joinpath(dataset_dir, "sparse", "0", "points3D.bin")
    images_dir = joinpath(dataset_dir, "images")
    ColmapDataset(;
        cameras_file, images_file, points_file,
        scale, images_dir, holdout, max_extent, with_thumbnails,
        use_masks, carve_with_masks, carve_tolerance)
end

"""
- `holdout`: the train/test split the 3DGS literature evaluates on (`llffhold`):
    views ordered by image filename, every `holdout`-th one held out for testing,
    the rest used for training.
    Deterministic, so the split matches other implementations view-for-view.
    `0` disables the split & trains on every view.
- `max_extent`: upper bound on the camera extent, which scales the position
    learning rate & the densification size thresholds.
    The reference implementation does not clamp it (pass `Inf32` to match).
- `with_thumbnails`: also downscale every train image to `THUMBNAIL_WIDTH` (`train_thumbnails`).
- `use_masks`: read the coverage masks of `masks/` (see `masking.jl`).
    `false` ignores the directory as if it were not there: no carve, no dropped views, no masked targets.
- `carve_with_masks`: drop init points the coverage masks place off the subject (see [`carve_points`](@ref)).
    Inert without a `masks/` directory.
    `carve_tolerance` is the fraction of the views seeing a point that may disagree before it is kept anyway.
"""
function ColmapDataset(;
    cameras_file::String, images_file::String, points_file::String,
    scale::Int = 1, images_dir::String, holdout::Int = 8,
    max_extent::Float32 = Inf32, with_thumbnails::Bool = false,
    use_masks::Bool = true, carve_with_masks::Bool = true,
    carve_tolerance::Float32 = 0.1f0,
)
    images_dir = scale > 1 ? "$(images_dir)_$(scale)" : images_dir
    depths_dir = joinpath(dirname(images_dir), "depths")
    has_depth_dir = isdir(depths_dir)
    sky_dir = joinpath(dirname(images_dir), "sky")
    has_sky_dir = isdir(sky_dir)
    masks_dir = joinpath(dirname(images_dir), "masks")
    has_masks_dir = use_masks && isdir(masks_dir)
    !use_masks && isdir(masks_dir) &&
        @info "Coverage masks are disabled: ignoring `$masks_dir`."

    colmap_cameras = NU.COLMAP.load_cameras_data(cameras_file)
    images = NU.COLMAP.load_images_data(images_file)
    points = NU.COLMAP.load_points_data(points_file)

    # NOTE: All cameras share intrinsics.
    cam = colmap_cameras[1]
    width, height = cam.resolution
    fx, fy, cx, cy = cam.intrinsics

    focal = SVector{2, Float32}(fx, fy) ./ Float32(scale)
    resolution = round.(UInt32, SVector{2, Float32}(width, height) ./ Float32(scale))
    principal = (SVector{2, Float32}(cx, cy) ./ Float32(scale)) ./ SVector{2, Float32}(resolution)

    # NOTE: no distortion.
    intrinsics = NU.CameraIntrinsics(nothing, focal, principal, resolution)
    @info "Dataset resolution: ($(Int(resolution[1])) x $(Int(resolution[2]))) (width x height)."
    with_thumbnails && @info "Creating thumbnails for the dataset, `width = $THUMBNAIL_WIDTH`."

    # Resolve the cameras & the per-view file paths.
    camera_centers = SVector{3, Float32}[]
    cameras = Camera[]
    depth_paths = Maybe{String}[]
    sky_paths = Maybe{String}[]
    mask_paths = Maybe{String}[]
    empty_masks = 0

    image_filenames = String[]
    thumbnails_list = Array{UInt8, 3}[]
    for (id, img) in images
        image_path = joinpath(images_dir, img.name)
        if !isfile(image_path)
            @info "Image files does not exist, skipping: `$image_path`."
            continue
        end

        # A mask that keeps nothing leaves an all-black target,
        # so the view is dropped rather than trained on.
        mask_path = has_masks_dir ? sidecar_path(masks_dir, img.name) : nothing
        if mask_path ≢ nothing && mask_is_empty(load_mask_raw(mask_path))
            empty_masks += 1
            continue
        end

        R = SMatrix{3, 3, Float32, 9}(QuatRotation(img.q...))
        t = SVector{3, Float32}(img.t...)
        cam = Camera(R, t; intrinsics, img_name=img.name)

        push!(camera_centers, cam.camera_center)
        push!(cameras, cam)
        push!(image_filenames, img.name)

        with_thumbnails && push!(thumbnails_list, thumbnail(load(image_path), THUMBNAIL_WIDTH))

        push!(depth_paths, has_depth_dir ? sidecar_path(depths_dir, img.name) : nothing)
        push!(sky_paths, has_sky_dir ? sidecar_path(sky_dir, img.name) : nothing)
        push!(mask_paths, mask_path)
    end
    empty_masks > 0 && @warn "Dropped $empty_masks image(s): their mask keeps no pixels."
    @info "Found `$(length(image_filenames))` images."

    # Compute cameras extent which is used for scaling learning rate and densification.
    scene_center = sum(camera_centers) ./ length(camera_centers)
    scene_diagonal = maximum(map(c -> norm(c - scene_center), camera_centers))
    scene_radius::Float32 = scene_diagonal * 1.1f0
    camera_extent::Float32 = min(max_extent, scene_radius)

    @info "Scene extent: $(round(camera_extent; digits=3))" * (
        camera_extent < scene_radius ?
            " (clamped from $(round(scene_radius; digits=3)) by `max_extent`)." :
            ".")

    # Remove points from initial point-cloud that are not covered by the masks.
    points_3d = Float32.(points.points_3d)
    points_colors = Float32.(points.points_colors) .* (1f0 / 255f0)
    if carve_with_masks && any(!isnothing, mask_paths)
        keep = carve_points(points_3d, cameras, mask_paths; tolerance=carve_tolerance)
        n_keep = count(keep)
        if n_keep < MIN_CARVED_POINTS
            @warn "Coverage masks would carve the init cloud down to " *
                "`$n_keep` point(s) — keeping it whole. Check that the masks " *
                "match their images & that the poses are the ones they were drawn on."
        else
            @info "Coverage masks carved the init cloud: `$(size(points_3d, 2))` → `$n_keep` points."
            points_3d = points_3d[:, keep]
            points_colors = points_colors[:, keep]
        end
    end

    scales = compute_scales(points_3d)

    # Views in filename order.
    order = sortperm(image_filenames)
    train_ids, test_ids = if holdout > 0
        [id for (i, id) in enumerate(order) if (i - 1) % holdout != 0],
        order[1:holdout:end]
    else
        order, Int[]
    end

    train_cameras = cameras[train_ids]
    train_thumbnails = with_thumbnails ? thumbnails_list[train_ids] : Array{UInt8, 3}[]
    train_image_filenames = image_filenames[train_ids]
    train_depth_paths = depth_paths[train_ids]
    train_sky_paths = sky_paths[train_ids]
    train_mask_paths = mask_paths[train_ids]

    test_cameras = cameras[test_ids]
    test_image_filenames = image_filenames[test_ids]
    test_mask_paths = mask_paths[test_ids]

    n_train_depths = count(!isnothing, train_depth_paths)
    n_train_depths > 0 && @info "Found depth priors for $n_train_depths / $(length(train_depth_paths)) train images."

    n_train_sky = count(!isnothing, train_sky_paths)
    n_train_sky > 0 && @info "Found sky masks for $n_train_sky / $(length(train_sky_paths)) train images."

    n_masks = count(!isnothing, mask_paths)
    n_masks > 0 && @info "Found masks for $n_masks / $(length(mask_paths)) images."

    ColmapDataset(
        points_3d,
        points_colors,
        scales,
        images_dir, resolution,
        train_image_filenames, train_cameras, train_thumbnails,
        train_depth_paths, n_train_depths > 0,
        has_depth_dir ? depths_dir : nothing,
        train_sky_paths, n_train_sky > 0, has_sky_dir ? sky_dir : nothing,
        train_mask_paths, test_mask_paths, n_masks > 0,
        has_masks_dir ? masks_dir : nothing,
        test_image_filenames, test_cameras,
        camera_extent)
end

function sidecar_path(sidecar_dir::String, image_name::String)
    path = joinpath(sidecar_dir, "$(splitext(image_name)[1]).png")
    return isfile(path) ? path : nothing
end

# Downscale a loaded (`height × width`) image to at most `max_width` pixels wide.
function thumbnail(image::AbstractMatrix, max_width::Int)
    height, width = size(image)
    if width > max_width
        scale = max_width / width
        image = imresize(image, (max(1, round(Int, height * scale)), max_width))
    end
    raw = floor.(UInt8, clamp.(Float32.(channelview(image)), 0f0, 1f0) .* 255f0)
    return permutedims(raw, (1, 3, 2))
end

function compute_scales(xyz::Matrix{Float32}; point_size::Float32 = 1f0)
    # Estimate initial covariance as an isotropic Gaussian with axes equal
    # to log of the mean of the distance to the closest 3 neighbors.
    kdtree = KDTree(xyz)
    _, dists = knn(kdtree, xyz, 3 + 1, true #= sort results =#)

    n_points = size(xyz, 2)
    scales = Matrix{Float32}(undef, 3, n_points)
    for (i, d) in enumerate(dists)
        md = mean(@view(d[2:end]).^2)
        scales[:, i] .= log(sqrt(max(1f-7, md) * point_size))
    end
    return scales
end

Base.length(d::ColmapDataset) = length(d.train_cameras)

"""
Everything view `idx` is supervised against.

`image` is `(channels, width, height)` `UInt8`,
the maps are `(width, height)` `Float32` at the same resolution,
`nothing` where the view has no such file.
"""
struct ViewData
    image::Array{UInt8, 3}
    mask::Maybe{Matrix{Float32}}
    sky_mask::Maybe{Matrix{Float32}}
    depth::Maybe{Matrix{Float32}}
    # Quantization step of `depth`'s encoding; `0f0` when there is no prior.
    depth_qstep::Float32
end

view_image_path(dataset::ColmapDataset, idx::Int, set::Symbol) = joinpath(
    dataset.images_dir, set == :train ?
        dataset.train_image_filenames[idx] : dataset.test_image_filenames[idx])

"""
The dataset's resolution as `(width, height)` `Int`s.
"""
view_resolution(dataset::ColmapDataset) = (Int(dataset.resolution[1]), Int(dataset.resolution[2]))

"""
Resize image to a `target` resolution if needed.
"""
fit_resolution(image::AbstractMatrix, target::Tuple{Int, Int}) =
    size(image) == target ? image : imresize(image, target)

"""
Decode a train / test image into the `(channels, width, height)` `UInt8` layout
the loss is computed from, at the dataset's resolution.
"""
function load_image(dataset::ColmapDataset, idx::Int, set::Symbol)
    image = load(view_image_path(dataset, idx, set))
    image = fit_resolution(image, reverse(view_resolution(dataset)))
    return permutedims(image_bytes(channelview(image)), (1, 3, 2))
end

image_bytes(channels::AbstractArray{N0f8}) = reinterpret(UInt8, channels)
image_bytes(channels::AbstractArray) = floor.(UInt8, Float32.(channels) .* 255f0)

"""
Load view `idx` of `set` in full. The four files are read on separate tasks:
they are independent, and at full resolution the image alone is ~500 ms of
decode & resize (see [`ViewLoader`](@ref)).

Only train views have depth priors & sky masks; coverage masks apply to both
splits, so training & scoring see the same image (`masking.jl`).
"""
function load_view(dataset::ColmapDataset, idx::Int, set::Symbol)
    width, height = view_resolution(dataset)

    mask_paths = set == :train ? dataset.train_mask_paths : dataset.test_mask_paths
    mask_path = mask_paths[idx]
    sky_path = set == :train ? dataset.train_sky_paths[idx] : nothing
    depth_path = set == :train ? dataset.train_depth_paths[idx] : nothing

    image_task = Threads.@spawn load_image(dataset, idx, set)
    mask_task = Threads.@spawn(mask_path ≡ nothing ? nothing : load_mask(mask_path, width, height))
    sky_task = Threads.@spawn(sky_path ≡ nothing ? nothing : load_mask(sky_path, width, height))
    depth_task = Threads.@spawn(depth_path ≡ nothing ? nothing : load_depth_prior(depth_path, width, height))

    depth_prior = fetch(depth_task)::Maybe{Tuple{Matrix{Float32}, Float32}}
    return ViewData(
        fetch(image_task)::Array{UInt8, 3},
        fetch(mask_task)::Maybe{Matrix{Float32}},
        fetch(sky_task)::Maybe{Matrix{Float32}},
        depth_prior ≡ nothing ? nothing : depth_prior[1],
        depth_prior ≡ nothing ? 0f0 : depth_prior[2])
end

device_image(kab, image::Array{UInt8, 3}) = adapt(kab, image) .* (1f0 / 255f0)

# From the `(c, w, h)` `UInt8` decode layout to the `(w, h, c, 1)` `Float32` one
# the loss & the edge map are computed in.
function device_target(kab, image::Array{UInt8, 3})
    image = device_image(kab, image)
    image = permutedims(image, (2, 3, 1))
    return reshape(image, size(image)..., 1)
end

get_image(dataset::ColmapDataset, kab, idx::Int, set::Symbol) =
    device_image(kab, load_image(dataset, idx, set))

"""
Reads the views the train loop is about to want, on background tasks,
so the disk work overlaps the GPU work.
"""
mutable struct ViewLoader
    dataset::ColmapDataset
    lookahead::Int
    # Train view index → the task reading it.
    pending::Dict{Int, Task}
end

ViewLoader(dataset::ColmapDataset; lookahead::Int = VIEW_LOOKAHEAD) =
    ViewLoader(dataset, lookahead, Dict{Int, Task}())

"""
Fetch `ViewData` either immediately (for :test set) or from a pre-fetching task (:train).
"""
function take_view!(loader::ViewLoader, idx::Int, set::Symbol)
    # Load :test immediately.
    set == :train || return load_view(loader.dataset, idx, set)

    # Try fetching pre-fetching task, load immediately if no such task.
    # Otherwise, wait on it.
    task = pop!(loader.pending, idx, nothing)
    task ≡ nothing && return load_view(loader.dataset, idx, :train)
    return fetch(task)::ViewData
end

"""
Queue upcoming `ViewData` for the training loop.

Remove views from queue that are not in `upcoming` list.
If views from the `upcoming` are already in queue, they arer left intact,
only new ones are added.
"""
function prefetch!(loader::ViewLoader, upcoming)
    filter!(kv -> kv.first in upcoming, loader.pending)
    for idx in upcoming
        haskey(loader.pending, idx) && continue
        loader.pending[idx] = Threads.@spawn load_view(loader.dataset, idx, :train)
    end
    return
end
