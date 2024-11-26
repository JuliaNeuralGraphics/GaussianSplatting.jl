# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
mutable struct Camera
    R::SMatrix{3, 3, Float32, 9}
    t::SVector{3, Float32}
    intrinsics::NU.CameraIntrinsics

    w2c::SMatrix{4, 4, Float32, 16}
    c2w::MMatrix{4, 4, Float32, 16}
    projection::SMatrix{4, 4, Float32, 16}
    full_projection::SMatrix{4, 4, Float32, 16}
    camera_center::SVector{3, Float32}

    original_focal::SVector{2, Float32}
    original_resolution::SVector{2, UInt32} # (w, h)

    img_name::String
end

function Camera(
    R::SMatrix{3, 3, Float32, 9}, t::SVector{3, Float32};
    intrinsics::NU.CameraIntrinsics, img_name::String,
)
    znear, zfar = 1f-2, 100f0
    fov_xy = NU.focal2fov.(intrinsics.resolution, intrinsics.focal)
    projection = NGL.perspective(fov_xy..., znear, zfar; zsign=1f0)

    w2c, c2w = get_w2c(R, t)
    full_projection = projection * w2c
    camera_center = c2w[1:3, 4]

    Camera(
        R, t, intrinsics, w2c, c2w,
        projection, full_projection, camera_center,
        intrinsics.focal, intrinsics.resolution, img_name)
end

function Camera(; fx::Float32, fy::Float32, width::Int, height::Int)
    focal = SVector{2, Float32}(fx, fy)
    principal = SVector{2, Float32}(0.5f0, 0.5f0)
    resolution = SVector{2, UInt32}(width, height)
    intrinsics = NU.CameraIntrinsics(nothing, focal, principal, resolution)
    Camera(
        SMatrix{3, 3, Float32, 9}(I), zeros(SVector{3, Float32});
        intrinsics, img_name="")
end

function set_resolution!(c::Camera; width::Int, height::Int)
    scale::Float32 = height / c.original_resolution[2]
    resolution = SVector{2, UInt32}(width, height)
    focal = c.original_focal .* scale
    c.intrinsics = NU.CameraIntrinsics(c.intrinsics; resolution, focal)
    c |> _update_from_intrinsics!
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
    c |> _update_from_c2w!
end

function shift!(c::Camera, relative)
    c.c2w[1:3, 4] .+= @view(c.c2w[1:3, 1:3]) * relative
    c |> _update_from_c2w!
end

function rotate!(c::Camera, rotation)
    c.c2w[1:3, 1:3] .= rotation * @view(c.c2w[1:3, 1:3])
    c |> _update_from_c2w!
end

function set_c2w!(c::Camera, c2w)
    c.c2w .= c2w
    c |> _update_from_c2w!
end

function _update_from_c2w!(c::Camera)
    w2c = inv(c.c2w)
    c.w2c, _ = get_w2c(
        SMatrix{3, 3, Float32}(w2c[1:3, 1:3]),
        SVector{3, Float32}(w2c[1:3, 4]))

    c.full_projection = c.projection * c.w2c
    c.camera_center = c.c2w[1:3, 4]
    return
end

function _update_from_intrinsics!(c::Camera)
    znear, zfar = 1f-2, 100f0
    fov_xy = NU.focal2fov.(c.intrinsics.resolution, c.intrinsics.focal)
    c.projection = NGL.perspective(fov_xy..., znear, zfar; zsign=1f0)
    c.full_projection = c.projection * c.w2c
    return
end

function rotation_6d_to_matrix(θ)
    a1, a2 = θ[1:3], θ[4:6]
    b1 = normalize(a1)
    b2 = a2 .- b1 .* sum(b1 .* a2)
    b3 = normalize(b2)
    b4 = b1 × b3
    return transpose(hcat(b1, b3, b4))
end
