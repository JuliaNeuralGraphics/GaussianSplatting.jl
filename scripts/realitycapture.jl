# RealityCapture / RealityScan scene conversion.
#
# RC exports a scene as an `input/` image directory, a `camera-params.csv` pose
# table & a dense `points.ply`. Two things keep it from being read directly:
# its intrinsics are per-image, and its radial distortion is large enough to
# matter (tens of pixels at the corners), while the training path assumes one
# shared pinhole over pre-undistorted images (see the notes in `dataset.jl`).
#
# `rc_convert` resolves both once & writes the COLMAP layout `ColmapDataset`
# already reads, the same way `gs-convert.jl` hands COLMAP's own undistorter
# the job. Included by `rc-convert.jl`; standalone, so it does not need the
# package loaded.

using FileIO
using ImageCore
using ImageTransformations
using LinearAlgebra
using Random
using Rotations
using StaticArrays
using Statistics

import PlyIO
import ProgressMeter

const RC_IMAGES_DIR = "input"
const RC_PARAMS_FILE = "camera-params.csv"
const RC_POINTS_FILE = "points.ply"

# Views whose focal deviates from the dataset's median by more than this are
# dropped: warping a zoomed-in frame into the shared pinhole would leave a
# black border around it, and that border would be trained against.
const RC_FOCAL_TOLERANCE = 0.02

"""
One row of `camera-params.csv`.

`position` is the camera center in RC's right-handed, Z-up world frame,
`ypr` are degrees, and the distortion coefficients follow Brown's model
(`radial` is `k1..k4`, `tangential` is `t1, t2`).
"""
struct RCView
    name::String
    position::SVector{3, Float64}
    ypr::SVector{3, Float64}
    f_35mm::Float64
    principal_norm::SVector{2, Float64}
    radial::SVector{4, Float64}
    tangential::SVector{2, Float64}
end

"""
Read `camera-params.csv`.

Columns are resolved by header name, not by position: RC's export templates are
user-editable, so the column order & the set of optional columns both vary.
Only the pose & `f_35mm` columns are required; a missing distortion column
means "no distortion", which is what RC writes for an ideal lens.
"""
function read_rc_params(csv_path::String)
    lines = readlines(csv_path)
    isempty(lines) && error("RealityCapture params file is empty: `$csv_path`.")

    header = [String(strip(lstrip(strip(c), '#'))) for c in split(lines[1], ',')]
    function find_column(names...)
        for name in names
            column = findfirst(==(name), header)
            column ≡ nothing || return column
        end
        return nothing
    end
    function required_column(names...)
        column = find_column(names...)
        column ≡ nothing && error(
            "`$csv_path` has no `$(names[1])` column. Header: $(join(header, ", ")).")
        column
    end

    name_col = required_column("name")
    x_col, y_col, z_col =
        required_column("x"), required_column("y"), required_column("alt", "z")
    yaw_col = required_column("yaw", "heading")
    pitch_col, roll_col = required_column("pitch"), required_column("roll")
    f_col = required_column("f_35mm", "f")
    px_col, py_col = find_column("px_norm", "px"), find_column("py_norm", "py")
    k_cols = ntuple(i -> find_column("k$i"), 4)
    t_cols = ntuple(i -> find_column("t$i", "p$i"), 2)

    views = RCView[]
    for (i, line) in enumerate(@view lines[2:end])
        isempty(strip(line)) && continue
        fields = split(line, ',')
        length(fields) < length(header) && error(
            "`$csv_path` line $(i + 1) has $(length(fields)) field(s), " *
            "expected $(length(header)).")

        number(column, default = 0.0) =
            column ≡ nothing ? default : parse(Float64, strip(fields[column]))

        push!(views, RCView(
            String(strip(fields[name_col])),
            SVector{3, Float64}(number(x_col), number(y_col), number(z_col)),
            SVector{3, Float64}(number(yaw_col), number(pitch_col), number(roll_col)),
            number(f_col),
            SVector{2, Float64}(number(px_col), number(py_col)),
            SVector{4, Float64}(number.(k_cols)),
            SVector{2, Float64}(number.(t_cols))))
    end
    isempty(views) && error("`$csv_path` lists no cameras.")
    return views
end

# RC's world frame is ENU (`x` east, `y` north, `alt` up) while its yaw/pitch/roll
# are the aerospace angles of a NED body frame, so the axis swap below sits
# outside the Euler product. `RC_BODY_TO_CAM` is `Rz(90°)`, taking that body
# frame to the OpenCV camera frame the rasterizer works in (+x right, +y down,
# +z forward — see `pos_world_to_cam` in `rasterization/projection.jl`).
#
# NOTE: a level RC camera has `pitch ≈ 90°`, which leaves `yaw` & `roll` almost
# gimbal-locked against each other. Reading the three angles as a ZXZ rotation
# instead — the intuitive "azimuth, elevation, tilt" — looks right on frames
# with little roll & silently falls apart on the rest.
const RC_ENU_TO_NED = SMatrix{3, 3, Float64, 9}(0, 1, 0, 1, 0, 0, 0, 0, -1)
const RC_BODY_TO_CAM = SMatrix{3, 3, Float64, 9}(0, 1, 0, -1, 0, 0, 0, 0, 1)

"""
The view's camera-to-world rotation.
"""
function rc_c2w(view::RCView)
    yaw, pitch, roll = deg2rad.(view.ypr)
    return RC_ENU_TO_NED * RotZ(yaw) * RotY(pitch) * RotX(roll) * RC_BODY_TO_CAM
end

"""
The view's pose in COLMAP's convention: `R` maps world to camera, `t` is the
world origin in camera space.
"""
function rc_pose(view::RCView)
    R = transpose(rc_c2w(view))
    return R, -R * view.position
end

"""
The view's pinhole intrinsics in pixels.

RC normalizes both the 35mm-equivalent focal & the principal point offset by the
*larger* image dimension, which is easy to miss on a portrait image.
"""
function rc_intrinsics(view::RCView, width::Int, height::Int)
    normalizer = max(width, height)
    return (;
        f = view.f_35mm / 36 * normalizer,
        cx = width / 2 + view.principal_norm[1] * normalizer,
        cy = height / 2 + view.principal_norm[2] * normalizer)
end

"""
Brown's distortion model, mapping ideal normalized image coordinates to the
distorted ones RC's coefficients are expressed against.
"""
@inline function rc_distort(view::RCView, xu::Float64, yu::Float64)
    k1, k2, k3, k4 = view.radial
    t1, t2 = view.tangential
    r² = xu * xu + yu * yu
    radial = 1.0 + r² * (k1 + r² * (k2 + r² * (k3 + r² * k4)))
    return (
        xu * radial + 2.0 * t1 * xu * yu + t2 * (r² + 2.0 * xu * xu),
        yu * radial + t1 * (r² + 2.0 * yu * yu) + 2.0 * t2 * xu * yu)
end

"""
The smallest focal at which every pixel of `view`'s undistorted image samples
from inside its source image.

The undistorted images share one pinhole, centered principal point included, so
a view both distorts & shifts on its way there. Rather than pad the result with
black, the shared focal is raised until the crop is clean — the `blank_pixels = 0`
behaviour of `colmap image_undistorter`.
"""
function rc_required_focal(view::RCView, width::Int, height::Int; samples::Int = 256)
    (; f, cx, cy) = rc_intrinsics(view, width, height)
    center_x, center_y = width / 2, height / 2

    # The warp is radial & monotone, so an output border that samples from
    # inside the source implies the interior does too.
    border = NTuple{2, Float64}[]
    for i in 0:samples
        s = i / samples
        push!(border, (s * width, 0.0), (s * width, Float64(height)))
        push!(border, (0.0, s * height), (Float64(width), s * height))
    end

    fits(focal) = all(border) do (x, y)
        xd, yd = rc_distort(view, (x - center_x) / focal, (y - center_y) / focal)
        source_x, source_y = cx + f * xd, cy + f * yd
        0.0 ≤ source_x ≤ width && 0.0 ≤ source_y ≤ height
    end

    hi = f
    while !fits(hi)
        hi *= 1.25
        hi > f * 16.0 && error(
            "`$(view.name)` cannot be undistorted into a pinhole: " *
            "its distortion coefficients are extreme.")
    end
    lo = hi / 2
    while fits(lo)
        lo /= 2
        lo < f * 1f-3 && break
    end
    for _ in 1:48
        mid = (lo + hi) / 2
        fits(mid) ? (hi = mid) : (lo = mid)
    end
    return hi
end

@inline function sample_bilinear(source::Array{Float32, 3}, x::Float64, y::Float64)
    height, width, _ = size(source)
    (1.0 ≤ x ≤ width && 1.0 ≤ y ≤ height) || return (0f0, 0f0, 0f0)

    x0, y0 = floor(Int, x), floor(Int, y)
    x1, y1 = min(x0 + 1, width), min(y0 + 1, height)
    dx, dy = Float32(x - x0), Float32(y - y0)
    @inbounds ntuple(3) do c
        top = source[y0, x0, c] * (1f0 - dx) + source[y0, x1, c] * dx
        bottom = source[y1, x0, c] * (1f0 - dx) + source[y1, x1, c] * dx
        top * (1f0 - dy) + bottom * dy
    end
end

"""
Resample `image` from `view`'s own distorted intrinsics into the shared pinhole
`(focal, width/2, height/2)`, keeping the resolution.
"""
function rc_undistort(image::AbstractMatrix, view::RCView, focal::Float64)
    height, width = size(image)
    (; f, cx, cy) = rc_intrinsics(view, width, height)
    center_x, center_y = width / 2, height / 2

    source = permutedims(Float32.(channelview(RGB{Float32}.(image))), (2, 3, 1))
    undistorted = Matrix{RGB{N0f8}}(undef, height, width)
    @inbounds for v in 1:height
        yu = (v - 0.5 - center_y) / focal
        for u in 1:width
            xd, yd = rc_distort(view, (u - 0.5 - center_x) / focal, yu)
            r, g, b = sample_bilinear(source, cx + f * xd + 0.5, cy + f * yd + 0.5)
            undistorted[v, u] = RGB{N0f8}(
                clamp(r, 0f0, 1f0), clamp(g, 0f0, 1f0), clamp(b, 0f0, 1f0))
        end
    end
    return undistorted
end

"""
Read the RC point cloud as `(points, colors)`, `3 × N` each,
world-space coordinates & `UInt8` colors.
"""
function read_rc_points(ply_path::String)
    ply = PlyIO.load_ply(ply_path)
    vertex = ply["vertex"]
    properties = Set(PlyIO.plyname.(vertex.properties))

    for axis in ("x", "y", "z")
        axis in properties || error(
            "`$ply_path` has no `$axis` vertex property. " *
            "Export the point cloud with XYZ & RGB.")
    end
    color_names = if all(c -> c in properties, ("red", "green", "blue"))
        ("red", "green", "blue")
    elseif all(c -> c in properties, ("diffuse_red", "diffuse_green", "diffuse_blue"))
        ("diffuse_red", "diffuse_green", "diffuse_blue")
    else
        error("`$ply_path` has no vertex colors. Export the point cloud with RGB.")
    end

    points = Matrix{Float64}(undef, 3, length(vertex["x"]))
    for (i, axis) in enumerate(("x", "y", "z"))
        points[i, :] .= vertex[axis]
    end
    colors = Matrix{UInt8}(undef, 3, size(points, 2))
    for (i, channel) in enumerate(color_names)
        colors[i, :] .= vertex[channel]
    end
    return points, colors
end

# COLMAP binary model writers.
#
# These are read straight back by `NerfUtils.COLMAP.load_*_data`, so they have
# to match that reader field for field — including the camera id, which
# `dataset.jl` looks up as `colmap_cameras[1]` rather than taking the first
# entry.
const COLMAP_CAMERA_ID = Int32(1)
const COLMAP_PINHOLE = Int32(1)

function write_colmap_cameras(path::String, width::Int, height::Int, focal::Float64)
    open(path, "w") do io
        write(io, UInt64(1))
        write(io, COLMAP_CAMERA_ID, COLMAP_PINHOLE, Int64(width), Int64(height))
        write(io, focal, focal, width / 2, height / 2)
    end
end

function write_colmap_images(path::String, views::Vector{RCView})
    io = IOBuffer()
    write(io, UInt64(length(views)))
    for (id, view) in enumerate(views)
        R, t = rc_pose(view)
        q = Rotations.params(QuatRotation(R)) # (w, x, y, z), COLMAP's order.

        write(io, Int32(id))
        write(io, Float64.(q)...)
        write(io, Float64.(t)...)
        write(io, COLMAP_CAMERA_ID)
        write(io, codeunits(view.name), UInt8(0))
        write(io, UInt64(0)) # No 2D observations: nothing downstream reads them.
    end
    write(path, take!(io))
    return
end

function write_colmap_points(path::String, points::Matrix{Float64}, colors::Matrix{UInt8})
    io = IOBuffer()
    write(io, UInt64(size(points, 2)))
    for i in axes(points, 2)
        write(io, UInt64(i))
        write(io, points[1, i], points[2, i], points[3, i])
        write(io, colors[1, i], colors[2, i], colors[3, i])
        write(io, 0.0)       # Reprojection error, unused by the dataset.
        write(io, UInt64(0)) # Empty track.
    end
    write(path, take!(io))
    return
end

focal2fov(resolution, focal) = 2 * rad2deg(atan(resolution / (2 * focal)))

"""
    rc_convert(source_dir; kwargs...)

Convert a RealityCapture / RealityScan export into the COLMAP layout
`ColmapDataset` reads: undistorted `images/` & a
`sparse/0/{cameras,images,points3D}.bin` model.

`source_dir` is expected to hold `$RC_IMAGES_DIR/`, `$RC_PARAMS_FILE` &
`$RC_POINTS_FILE`.

- `output_dir`: where the conversion is written, `source_dir` by default.
- `resize`: also write 1/2, 1/4 & 1/8 scale image directories.
- `max_points`: randomly subsample the init cloud down to this many points.
    `0` keeps it whole. RC clouds carry far-field background the cameras never
    get near, so capping is sometimes worth it.
- `focal_tolerance`: drop views whose focal deviates from the median by more
    than this fraction (see `RC_FOCAL_TOLERANCE`).
"""
function rc_convert(source_dir::String;
    output_dir::String = source_dir, resize::Bool = false,
    max_points::Int = 0, focal_tolerance::Float64 = RC_FOCAL_TOLERANCE,
)
    images_dir = joinpath(source_dir, RC_IMAGES_DIR)
    params_file = joinpath(source_dir, RC_PARAMS_FILE)
    points_file = joinpath(source_dir, RC_POINTS_FILE)
    for path in (images_dir, params_file, points_file)
        ispath(path) || error("RealityCapture export is missing `$path`.")
    end

    views = read_rc_params(params_file)
    @info "Read `$(length(views))` camera(s) from `$params_file`."

    # RC lists only the images it managed to register, and a capture usually has
    # more frames than that.
    n_listed = length(views)
    filter!(view -> isfile(joinpath(images_dir, view.name)), views)
    n_listed > length(views) && @warn(
        "Dropped $(n_listed - length(views)) camera(s): no such file in `$images_dir`.")
    isempty(views) && error("None of the cameras in `$params_file` have an image.")

    n_unregistered = count(readdir(images_dir)) do file
        isfile(joinpath(images_dir, file)) && !any(v -> v.name == file, views)
    end
    n_unregistered > 0 && @info(
        "`$images_dir` holds $n_unregistered file(s) with no camera — " *
        "unregistered frames & sidecars are ignored.")

    # A zoomed frame warped into the shared pinhole would be a small image in a
    # black frame, and the border would be trained against.
    median_focal = median(view.f_35mm for view in views)
    is_zoomed(view) = abs(view.f_35mm - median_focal) > focal_tolerance * median_focal
    zoomed = filter(is_zoomed, views)
    if !isempty(zoomed)
        @warn "Dropped $(length(zoomed)) view(s) whose focal deviates from the " *
            "median $(round(median_focal; digits=2))mm by more than " *
            "$(round(focal_tolerance * 100; digits=1))%: " *
            join(("$(v.name) ($(round(v.f_35mm; digits=2))mm)" for v in zoomed), ", ")
        filter!(!is_zoomed, views)
    end

    height, width = size(load(joinpath(images_dir, views[1].name)))
    @info "Image resolution: ($width x $height) (width x height)."

    focal = maximum(v -> rc_required_focal(v, width, height), views)
    @info "Undistorting into a shared pinhole: " *
        "focal $(round(focal; digits=2))px, " *
        "fov $(round(focal2fov(width, focal); digits=2))° x " *
        "$(round(focal2fov(height, focal); digits=2))°."

    out_images_dir = joinpath(output_dir, "images")
    mkpath(out_images_dir)
    scales = resize ? (2, 4, 8) : ()
    for scale in scales
        mkpath(joinpath(output_dir, "images_$scale"))
    end

    progress = ProgressMeter.Progress(length(views); desc="Undistorting: ")
    Threads.@threads for view in views
        undistorted = rc_undistort(load(joinpath(images_dir, view.name)), view, focal)
        save(joinpath(out_images_dir, view.name), undistorted)
        for scale in scales
            save(
                joinpath(output_dir, "images_$scale", view.name),
                imresize(undistorted, (height ÷ scale, width ÷ scale)))
        end
        ProgressMeter.next!(progress)
    end
    ProgressMeter.finish!(progress)

    points, colors = read_rc_points(points_file)
    @info "Read `$(size(points, 2))` init point(s) from `$points_file`."
    if max_points > 0 && size(points, 2) > max_points
        keep = randperm(size(points, 2))[1:max_points]
        points, colors = points[:, keep], colors[:, keep]
        @info "Subsampled the init cloud to `$max_points` points."
    end

    sparse_dir = joinpath(output_dir, "sparse", "0")
    mkpath(sparse_dir)
    write_colmap_cameras(joinpath(sparse_dir, "cameras.bin"), width, height, focal)
    write_colmap_images(joinpath(sparse_dir, "images.bin"), views)
    write_colmap_points(joinpath(sparse_dir, "points3D.bin"), points, colors)

    @info "Wrote `$(length(views))` image(s) & a COLMAP model to `$output_dir`."
    return output_dir
end
