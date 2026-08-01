# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
struct ColmapDataset{I <: AbstractArray{UInt8, 4}}
    points::Matrix{Float32}
    colors::Matrix{Float32}
    scales::Matrix{Float32}

    train_image_filenames::Vector{String}
    train_cameras::Vector{Camera}
    train_images::I

    # Monocular depth priors for depth supervision (`nothing` when an image has no prior)
    # + their encoding quantization steps.
    train_depths::Vector{Maybe{Matrix{Float32}}}
    train_depth_qsteps::Vector{Float32}
    has_depth_priors::Bool
    # Directory the depth priors were loaded from (`nothing` when absent);
    # the fitted-anchor sidecar cache lives next to it.
    depths_dir::Maybe{String}

    test_image_filenames::Vector{String}
    test_cameras::Vector{Camera}
    test_images::I

    camera_extent::Float32
end

function ColmapDataset(dataset_dir::String;
    scale::Int = 1, holdout::Int = 8, max_extent::Float32 = Inf32,
)
    cameras_file = joinpath(dataset_dir, "sparse", "0", "cameras.bin")
    images_file = joinpath(dataset_dir, "sparse", "0", "images.bin")
    points_file = joinpath(dataset_dir, "sparse", "0", "points3D.bin")
    images_dir = joinpath(dataset_dir, "images")
    ColmapDataset(;
        cameras_file, images_file, points_file,
        scale, images_dir, holdout, max_extent)
end

"""
- `holdout`: the train/test split the 3DGS literature evaluates on
  (`llffhold`): views ordered by image filename, every `holdout`-th one held
  out for testing, the rest used for training. Deterministic, so the split
  matches other implementations view-for-view. `0` disables the split & trains on every view.
- `max_extent`: upper bound on the camera extent, which scales the position
  learning rate & the densification size thresholds. The reference
  implementation does not clamp it (pass `Inf32` to match).
"""
function ColmapDataset(;
    cameras_file::String, images_file::String, points_file::String,
    scale::Int = 1, images_dir::String, holdout::Int = 8,
    max_extent::Float32 = Inf32,
)
    images_dir = scale > 1 ? "$(images_dir)_$(scale)" : images_dir
    depths_dir = joinpath(dirname(images_dir), "depths")
    has_depth_dir = isdir(depths_dir)

    colmap_cameras = NU.COLMAP.load_cameras_data(cameras_file)
    images = NU.COLMAP.load_images_data(images_file)
    points = NU.COLMAP.load_points_data(points_file)

    # NOTE: All cameras share intrinsics.
    cam = colmap_cameras[1]
    width, height = cam.resolution
    fx, fy, cx, cy = cam.intrinsics

    focal = SVector{2, Float32}(fx, fy) ./ Float32(scale)
    principal = SVector{2, Float32}(cx, cy) ./ SVector{2, Float32}(width, height)
    resolution = round.(UInt32, SVector{2, Float32}(width, height) ./ Float32(scale))
    # Round resolution to a multiple of 16.
    new_resolution = 16u32 .* cld.(resolution, 16u32)
    new_focal = Float32(new_resolution[2] / resolution[2]) .* focal

    # NOTE: no distortion.
    intrinsics = NU.CameraIntrinsics(nothing, new_focal, principal, new_resolution)

    # Load cameras and images & depths priors if they exist.
    camera_centers = SVector{3, Float32}[]
    cameras = Camera[]
    depth_priors_count = 0
    depth_maps = Maybe{Matrix{Float32}}[]
    depth_qsteps = Float32[]

    image_filenames = String[]
    images_list = Array{UInt8, 3}[]
    for (id, img) in images
        image_path = joinpath(images_dir, img.name)
        if !isfile(image_path)
            @info "Image files does not exist, skipping: `$image_path`."
            continue
        end

        R = SMatrix{3, 3, Float32, 9}(QuatRotation(img.q...))
        t = SVector{3, Float32}(img.t...)
        cam = Camera(R, t; intrinsics, img_name=img.name)

        push!(camera_centers, cam.camera_center)
        push!(cameras, cam)
        push!(image_filenames, img.name)

        # Round resolution to a multiple of 16, just like with cameras.
        image = load(image_path)
        image = imresize(image, reverse((Int.(new_resolution)...,)))

        raw = floor.(UInt8, Float32.(channelview(image)) .* 255f0)
        raw = permutedims(raw, (1, 3, 2))
        push!(images_list, raw)

        if has_depth_dir
            depth_path = joinpath(depths_dir, "$(splitext(img.name)[1]).png")
            depth, qstep = load_depth_prior(depth_path, Int(new_resolution[1]), Int(new_resolution[2]))
            push!(depth_maps, depth)
            push!(depth_qsteps, qstep)
            depth_priors_count += 1
        else
            push!(depth_maps, nothing)
            push!(depth_qsteps, 0f0)
        end
    end
    images = cat(images_list...; dims=4)

    # Compute cameras extent which is used for scaling learning rate and densification.
    scene_center = sum(camera_centers) ./ length(camera_centers)
    scene_diagonal = maximum(map(c -> norm(c - scene_center), camera_centers))
    scene_radius::Float32 = scene_diagonal * 1.1f0
    camera_extent::Float32 = min(max_extent, scene_radius)

    @info "Scene extent: $(round(camera_extent; digits=3))" * (
        camera_extent < scene_radius ?
            " (clamped from $(round(scene_radius; digits=3)) by `max_extent`)." :
            ".")

    scales = compute_scales(points.points_3d)

    # Views in filename order: the order the split is defined in & the one
    # other implementations report their per-view metrics in.
    order = sortperm(image_filenames)
    train_ids, test_ids = if holdout > 0
        [id for (i, id) in enumerate(order) if (i - 1) % holdout != 0],
        order[1:holdout:end]
    else
        order, Int[]
    end

    train_cameras = cameras[train_ids]
    train_images = images[:, :, :, train_ids]
    train_image_filenames = image_filenames[train_ids]
    train_depths = depth_maps[train_ids]
    train_depth_qsteps = depth_qsteps[train_ids]

    test_cameras = cameras[test_ids]
    test_images = images[:, :, :, test_ids]
    test_image_filenames = image_filenames[test_ids]

    depth_priors_count > 0 &&
        @info "Found depth priors for $depth_priors_count / $(length(train_depths)) train images."

    ColmapDataset(
        Float32.(points.points_3d),
        Float32.(points.points_colors) .* (1f0 / 255f0),
        scales,
        train_image_filenames, train_cameras, train_images,
        train_depths, train_depth_qsteps, depth_priors_count > 0,
        has_depth_dir ? depths_dir : nothing,
        test_image_filenames, test_cameras, test_images,
        camera_extent)
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

function get_image(dataset::ColmapDataset, kab, idx::Int, set::Symbol)
    image = if set == :train
        dataset.train_images[:, :, :, idx]
    else
        dataset.test_images[:, :, :, idx]
    end
    return adapt(kab, Float32.(image) .* (1f0 / 255f0))
end
