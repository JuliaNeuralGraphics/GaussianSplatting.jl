# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
struct ColmapDataset{
    P <: AbstractMatrix{Float32},
    C <: AbstractMatrix{Float32},
    I <: AbstractArray{UInt8, 4},
}
    points::P
    colors::C
    scales::P

    train_image_filenames::Vector{String}
    train_cameras::Vector{Camera}
    train_images::I

    test_image_filenames::Vector{String}
    test_cameras::Vector{Camera}
    test_images::I

    camera_extent::Float32
end

function ColmapDataset(kab,
    dataset_dir::String; scale::Int = 1, train_test_split::Real = 0.8,
)
    cameras_file = joinpath(dataset_dir, "sparse/0/cameras.bin")
    images_file = joinpath(dataset_dir, "sparse/0/images.bin")
    points_file = joinpath(dataset_dir, "sparse/0/points3D.bin")
    images_dir = joinpath(dataset_dir, "images")
    ColmapDataset(kab;
        cameras_file, images_file, points_file,
        scale, images_dir, train_test_split)
end

function ColmapDataset(kab;
    cameras_file::String, images_file::String, points_file::String,
    scale::Int = 1, images_dir::String, train_test_split::Real = 0.8,
)
    images_dir = scale > 1 ? "$(images_dir)_$(scale)" : images_dir

    colmap_cameras = NU.COLMAP.load_cameras_data(cameras_file)
    images = NU.COLMAP.load_images_data(images_file)
    points = NU.COLMAP.load_points_data(points_file)

    # Create intrinsics.
    cam = colmap_cameras[1] # All cameras share intrinsics.
    width, height = cam.resolution
    fx, fy, cx, cy = cam.intrinsics

    focal = SVector{2, Float32}(fx, fy) ./ Float32(scale)
    principal = SVector{2, Float32}(cx, cy) ./ SVector{2, Float32}(width, height)
    resolution = round.(UInt32, SVector{2, Float32}(width, height) ./ Float32(scale))
    # Round resolution to a multiple of 16.
    new_resolution = 16u32 .* cld.(resolution, 16u32)
    new_focal = Float32(new_resolution[2] / resolution[2]) .* focal

    intrinsics = NU.CameraIntrinsics(nothing, # TODO no distortion for now
        new_focal, principal, new_resolution)

    # Load cameras and images.
    camera_centers = SVector{3, Float32}[]
    cameras = Camera[]

    image_filenames = String[]
    images_list = Array{UInt8, 3}[]
    for (id, img) in images
        R = SMatrix{3, 3, Float32, 9}(QuatRotation(img.q...))
        t = SVector{3, Float32}(img.t...)
        cam = Camera(R, t; intrinsics, img_name=img.name)

        push!(camera_centers, cam.camera_center)
        push!(cameras, cam)

        push!(image_filenames, img.name)

        image = load(joinpath(images_dir, img.name))
        # Round resolution to a multiple of 16, just like with cameras.
        image = imresize(image, reverse((Int.(new_resolution)...,)))

        raw = floor.(UInt8, Float32.(channelview(image)) .* 255f0)
        raw = permutedims(raw, (1, 3, 2))
        push!(images_list, raw)
    end
    images = cat(images_list...; dims=4)

    # Compute cameras extent which is used for scaling learning rate
    # and densification.
    scene_center = sum(camera_centers) ./ length(camera_centers)
    scene_diagonal = maximum(map(c -> norm(c - scene_center), camera_centers))
    camera_extent::Float32 = scene_diagonal * 1.1f0

    scales = compute_scales(points.points_3d)

    n_cameras = length(cameras)
    perm = randperm(n_cameras)

    cameras = cameras[perm]
    images = images[:, :, :, perm]
    image_filenames = image_filenames[perm]

    if train_test_split < 1
        n_train = ceil(Int, n_cameras * train_test_split)

        train_cameras = cameras[1:n_train]
        train_images = images[:, :, :, 1:n_train]
        train_image_filenames = image_filenames[1:n_train]

        test_cameras = cameras[(n_train + 1):end]
        test_images = images[:, :, :, (n_train + 1):end]
        test_image_filenames = image_filenames[(n_train + 1):end]
    else
        train_cameras = cameras
        train_images = images
        train_image_filenames = image_filenames

        test_cameras = Camera[]
        test_images = Array{UInt8, 4}(undef, 0, 0, 0, 0)
        test_image_filenames = String[]
    end

    ColmapDataset(
        # TODO why move to GPU?
        adapt(kab, Float32.(points.points_3d)),
        adapt(kab, Float32.(points.points_colors) .* (1f0 / 255f0)),
        adapt(kab, scales),
        train_image_filenames, train_cameras, train_images,
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
