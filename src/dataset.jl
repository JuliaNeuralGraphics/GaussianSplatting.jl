mutable struct Camera
    R::SMatrix{3, 3, Float32, 9}
    t::SVector{3, Float32}
    intrinsics::NU.CameraIntrinsics

    w2c::SMatrix{4, 4, Float32, 16}
    c2w::MMatrix{4, 4, Float32, 16}
    projection::SMatrix{4, 4, Float32, 16}
    full_projection::SMatrix{4, 4, Float32, 16}
    camera_center::SVector{3, Float32}

    img_name::String
end

function Camera(
    R::SMatrix{3, 3, Float32, 9}, t::SVector{3, Float32};
    intrinsics::NU.CameraIntrinsics, img_name::String,
    znear::Float32 = 0.01f0, zfar::Float32 = 100f0,
)
    fov_xy = NU.focal2fov.(intrinsics.resolution, intrinsics.focal)
    projection = NGL.perspective(fov_xy..., znear, zfar; zsign=1f0)

    w2c, c2w = get_w2c(R, t)
    full_projection = projection * w2c
    camera_center = c2w[1:3, 4]

    Camera(
        R, t, intrinsics, w2c, c2w,
        projection, full_projection,
        camera_center, img_name)
end

resolution(c::Camera) = (;
    width=Int(c.intrinsics.resolution[1]),
    height=Int(c.intrinsics.resolution[2]))

view_dir(c::Camera) = SVector{3, Float32}(@view(c.c2w[1:3, 3])...)

view_up(c::Camera) = SVector{3, Float32}(@view(c.c2w[1:3, 2])...)

view_side(c::Camera) = SVector{3, Float32}(@view(c.c2w[1:3, 1])...)

view_pos(c::Camera) = SVector{3, Float32}(@view(c.c2w[1:3, 4])...)

look_at(c::Camera) = view_pos(c) .+ view_dir(c)

function get_w2c(R::SMatrix{3, 3, Float32, 9}, t::SVector{3, Float32})
    P = zeros(MMatrix{4, 4, Float32, 16})
    P[1:3, 1:3] .= R
    P[1:3, 4] .= t
    P[4, 4] = 1f0

    w2c = P
    c2w = inv(P)
    return w2c, c2w
end

function set_c2w!(c::Camera, R, t)
    c.c2w[1:3, 1:3] .= R
    c.c2w[1:3, 4] .= t
    _upd!(c)
end

function shift!(c::Camera, relative)
    c.c2w[1:3, 4] .+= @view(c.c2w[1:3, 1:3]) * relative
    _upd!(c)
end

function rotate!(c::Camera, rotation)
    c.c2w[1:3, 1:3] .= rotation * @view(c.c2w[1:3, 1:3])
    _upd!(c)
end

function _upd!(c::Camera)
    w2c = inv(c.c2w)
    c.w2c, _ = get_w2c(
        SMatrix{3, 3, Float32}(w2c[1:3, 1:3]),
        SVector{3, Float32}(w2c[1:3, 4]))

    c.full_projection = c.projection * c.w2c
    c.camera_center = c.c2w[1:3, 4]
    return
end

struct ColmapDataset{
    P <: AbstractMatrix{Float32},
    C <: AbstractMatrix{Float32},
    I <: AbstractArray{UInt8, 4},
}
    points::P
    colors::C
    scales::P
    cameras::Vector{Camera}
    images::I

    image_filenames::Vector{String}
end

function ColmapDataset(kab;
    cameras_file::String, images_file::String, points_file::String,
    scale::Int = 1, images_dir::String,
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
    image_filenames = String[]
    cameras = Camera[]
    images_list = Array{UInt8, 3}[]
    for (id, img) in images
        R = SMatrix{3, 3, Float32, 9}(QuatRotation(img.q...))
        t = SVector{3, Float32}(img.t...)
        push!(cameras, Camera(R, t; intrinsics, img_name=img.name))
        push!(image_filenames, img.name)

        image = load(joinpath(images_dir, img.name))
        # Round resolution to a multiple of 16, just like with cameras.
        image = imresize(image, reverse((Int.(new_resolution)...,)))

        raw = floor.(UInt8, Float32.(channelview(image)) .* 255f0)
        raw = permutedims(raw, (1, 3, 2))
        push!(images_list, raw)
    end
    images = cat(images_list...; dims=4)

    # Estimate initial covariance as an isotropic Gaussian with axes equal
    # to log of the mean of the distance to the closest 3 neighbors.
    kdtree = KDTree(points.points_3d)
    _, dists = knn(kdtree, points.points_3d, 3 + 1, true #= sort results =#)

    n_points = size(points.points_3d, 2)
    scales = Matrix{Float32}(undef, 3, n_points)
    for (i, d) in enumerate(dists)
        md = mean(@view(d[2:end]).^2)
        scales[:, i] .= log(sqrt(max(1f-7, md)))
    end

    ColmapDataset(
        adapt(kab, Float32.(points.points_3d)),
        adapt(kab, Float32.(points.points_colors) ./ 255f0),
        adapt(kab, scales),
        cameras, images, image_filenames)
end

Base.length(d::ColmapDataset) = length(d.cameras)

function get_image(dataset::ColmapDataset, kab, idx::Int)
    adapt(kab, Float32.(dataset.images[:, :, :, idx]) .* (1f0 / 255f0))
end
