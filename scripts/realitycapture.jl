# RealityCapture / RealityScan scene conversion.
#
# RC exports a scene as an `input/` image directory, a `camera-params.csv` pose
# & lens table, a `bundler.out` Bundler v0.3 reconstruction with its
# `images.txt` name list, and a `points.ply` cloud. Three things keep that from
# being read directly: the intrinsics are per-image, the radial distortion is
# large enough to matter (tens of pixels at the corners), and the training path
# assumes one shared pinhole over pre-undistorted images (see the notes in
# `dataset.jl`).
#
# `rc_convert` resolves all three once & writes the COLMAP layout
# `ColmapDataset` already reads, the same way `gs-convert.jl` hands COLMAP's own
# undistorter the job. Included by `rc-convert.jl`; standalone, so it does not
# need the package loaded.

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
const RC_POINTS_FILE = "points.ply"
const RC_BUNDLER_FILE = "bundler.out"
# RC has shipped the camera table & the Bundler name list under either name.
const RC_PARAMS_FILES = ("camera-params.csv", "cameras.csv")
const RC_BUNDLER_LISTS = ("images.txt", "list.txt")

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


# Bundler v0.3, RC's only export that carries SfM tracks — which point each view
# actually saw. `dataset.jl` resolves those into `train_visible_points` and
# `fit_depth_anchors` fits the depth anchors on them alone; without a track it
# falls back to `visible_by_zbuffer`, an approximation (see the warning in
# `regularizations/depth_supervision.jl`). So the poses, the points & the tracks
# all come from the `.out`.
#
# What the `.out` does *not* carry is the lens: RC writes `k1 = k2 = 0` there &
# puts the principal point at the image center, while the real lens needs the
# `k1..k4, t1, t2` of the CSV — 163px of uncorrected radial at the corner on a
# typical phone capture. The two agree on the focal, so the CSV stays the
# authority on distortion & the `.out` on geometry.

# Bundler's cameras look down `-z` with `+y` up; the rasterizer's look down `+z`
# with `+y` down, so a pose crosses over by a left-multiply with this. RC offers
# "Bundler v0.3" & "Bundler v0.3 negative z" as separate export options and the
# file records which one it was, so `bundler_flip` votes rather than assumes.
const BUNDLER_TO_COLMAP = SMatrix{3, 3, Float64, 9}(1, 0, 0, 0, -1, 0, 0, 0, -1)

# RC writes the Bundler cloud in a world frame `Rx(-90°)` away from the ENU
# frame of `camera-params.csv`: ENU `(x, y, alt)` lands at `(x, -alt, y)`. Only
# needed to fold an extra `points.ply` into the model, and checked against the
# camera centers before it is trusted (see `rc_enu_matches_bundler`).
const RC_ENU_TO_BUNDLER = SMatrix{3, 3, Float64, 9}(1, 0, 0, 0, 0, 1, 0, -1, 0)

"""
One `<camera> <key> <x> <y>` entry of a Bundler point's view list: the camera
that saw the point (0-based, indexing the `.out`'s own camera order) & where it
landed, in pixels from the image center with `+y` up.
"""
struct BundlerObservation
    camera::Int
    x::Float64
    y::Float64
end

struct BundlerPoint
    position::SVector{3, Float64}
    color::SVector{3, UInt8}
    views::Vector{BundlerObservation}
end

"""
One camera of a Bundler file. `R` maps world to camera & `t` is the world origin
in camera space, both in Bundler's own axis convention (see
[`BUNDLER_TO_COLMAP`](@ref)). An unregistered camera has `focal == 0`.
"""
struct BundlerCamera
    focal::Float64
    R::SMatrix{3, 3, Float64, 9}
    t::SVector{3, Float64}
end

# Invariant under `BUNDLER_TO_COLMAP`, which cancels with its own transpose.
bundler_center(camera::BundlerCamera) = -transpose(camera.R) * camera.t

"""
Read a Bundler v0.3 `.out` file as `(cameras, points)`, both in file order.
"""
function read_bundler(path::String)
    open(path, "r") do io
        header = readline(io)
        occursin("Bundle file", header) || error(
            "`$path` is not a Bundler file (first line: `$header`).")

        counts = split(strip(readline(io)))
        length(counts) == 2 || error(
            "`$path` has a malformed camera/point count line: `$(join(counts, " "))`.")
        n_cameras, n_points = parse.(Int, counts)

        # Bundler is strictly line-structured: 5 lines per camera, 3 per point.
        function numbers(n)
            fields = split(strip(readline(io)))
            length(fields) == n || error(
                "`$path` has a malformed line: expected $n value(s), " *
                "got $(length(fields)).")
            return parse.(Float64, fields)
        end

        cameras = Vector{BundlerCamera}(undef, n_cameras)
        for i in 1:n_cameras
            focal = numbers(3)[1] # `<f> <k1> <k2>`, and RC's k's are zeros.
            row1, row2, row3 = numbers(3), numbers(3), numbers(3)
            cameras[i] = BundlerCamera(focal,
                SMatrix{3, 3, Float64, 9}( # The rows above, laid out by column.
                    row1[1], row2[1], row3[1],
                    row1[2], row2[2], row3[2],
                    row1[3], row2[3], row3[3]),
                SVector{3, Float64}(numbers(3)))
        end

        points = Vector{BundlerPoint}(undef, n_points)
        for i in 1:n_points
            position = SVector{3, Float64}(numbers(3))
            color = SVector{3, UInt8}(clamp.(round.(Int, numbers(3)), 0, 255))

            fields = split(strip(readline(io)))
            n_views = round(Int, parse(Float64, fields[1]))
            length(fields) == 1 + 4 * n_views || error(
                "`$path` point $i lists $n_views view(s) but carries " *
                "$(length(fields) - 1) field(s) after the count.")
            views = Vector{BundlerObservation}(undef, n_views)
            for j in 1:n_views
                base = 1 + (j - 1) * 4
                views[j] = BundlerObservation(
                    round(Int, parse(Float64, fields[base + 1])),
                    parse(Float64, fields[base + 3]),
                    parse(Float64, fields[base + 4]))
            end
            points[i] = BundlerPoint(position, color, views)
        end
        return (; cameras, points)
    end
end

"""
Read the Bundler image list: one image per line, in camera order.

RC writes absolute Windows paths, so only the file name is kept. Bundler allows
an optional focal column after the name; trailing numbers are stripped rather
than splitting on the first space, so paths with spaces in them survive.
"""
function read_bundler_list(path::String)
    names = String[]
    for line in eachline(path)
        entry = strip(line)
        isempty(entry) && continue
        fields = split(entry)
        while length(fields) > 1 && tryparse(Float64, fields[end]) ≢ nothing
            fields = fields[1:end - 1]
        end
        push!(names, basename(replace(join(fields, " "), '\\' => '/')))
    end
    isempty(names) && error("`$path` names no images.")
    return names
end

"""
The left-multiply taking this export's poses into the rasterizer's convention:
[`BUNDLER_TO_COLMAP`](@ref) for a `-z` forward export, the identity for a `+z`
one.

RC's two Bundler options differ by exactly that factor & the file does not say
which was used, so the two are told apart by projecting the cloud through every
camera under both hypotheses and keeping whichever puts more of it inside the
frame. A real scene is a landslide for one of them: on a 228-view fountain the
flip wins 961k in-frame samples to 321k.
"""
function bundler_flip(bundler, width::Int, height::Int; samples::Int = 4096)
    stride = max(1, cld(length(bundler.points), samples))
    sampled = @view bundler.points[1:stride:end]

    in_frame(flip) = sum(bundler.cameras) do camera
        camera.focal > 0 || return 0
        R, t = flip * camera.R, flip * camera.t
        count(sampled) do point
            p = R * point.position + t
            p[3] > 0 || return false
            u = camera.focal * p[1] / p[3] + width / 2
            v = camera.focal * p[2] / p[3] + height / 2
            return 0 ≤ u ≤ width && 0 ≤ v ≤ height
        end
    end

    as_is = in_frame(one(SMatrix{3, 3, Float64, 9}))
    flipped = in_frame(BUNDLER_TO_COLMAP)
    @info "Bundler cameras look down $(flipped ≥ as_is ? "-z" : "+z") " *
        "($flipped in-frame sample(s) flipped vs $as_is as-is)."
    return flipped ≥ as_is ? BUNDLER_TO_COLMAP : one(SMatrix{3, 3, Float64, 9})
end

"""
Whether [`RC_ENU_TO_BUNDLER`](@ref) really maps this export's CSV positions onto
its Bundler camera centers, to within `tolerance` of the scene's own size.

The two clouds can only be mixed when it does, so this is checked rather than
assumed — a silently misaligned init cloud is far more expensive than skipping it.
"""
function rc_enu_matches_bundler(cameras, positions; tolerance::Float64 = 1e-3)
    centers = [bundler_center(camera) for camera in cameras]
    center = mean(centers)
    extent = maximum(norm(c - center) for c in centers)
    worst = maximum(zip(centers, positions)) do (c, position)
        norm(c - RC_ENU_TO_BUNDLER * position)
    end
    return worst ≤ tolerance * max(extent, eps())
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

"""
One image of the COLMAP model: its pose, its file name & the 2D observations
tying it to `points3D.bin`.
"""
struct ColmapImage
    name::String
    q::SVector{4, Float64} # (w, x, y, z), COLMAP's order.
    t::SVector{3, Float64}
    # `(x, y, point3D id)` per observation, in COLMAP's top-left pixel frame.
    observations::Vector{Tuple{Float64, Float64, UInt64}}
end

function write_colmap_images(path::String, images::Vector{ColmapImage})
    io = IOBuffer()
    write(io, UInt64(length(images)))
    for (id, image) in enumerate(images)
        write(io, Int32(id))
        write(io, image.q...)
        write(io, image.t...)
        write(io, COLMAP_CAMERA_ID)
        write(io, codeunits(image.name), UInt8(0))
        write(io, UInt64(length(image.observations)))
        for (x, y, point_id) in image.observations
            write(io, x, y, point_id)
        end
    end
    write(path, take!(io))
    return
end

"""
Write `points3D.bin`. `tracks[i]` holds the `(image id, observation index)`
pairs of point `i`; an empty one is allowed & means the point initializes the
model without taking part in the depth-anchor fit.
"""
function write_colmap_points(path::String,
    points::Vector{SVector{3, Float64}}, colors::Vector{SVector{3, UInt8}},
    tracks::Vector{Vector{Tuple{UInt32, UInt32}}},
)
    io = IOBuffer()
    write(io, UInt64(length(points)))
    for i in eachindex(points)
        write(io, UInt64(i))
        write(io, points[i]...)
        write(io, colors[i]...)
        write(io, 0.0) # Reprojection error, unused by the dataset.
        write(io, UInt64(length(tracks[i])))
        for (image_id, index) in tracks[i]
            write(io, image_id, index)
        end
    end
    write(path, take!(io))
    return
end

focal2fov(resolution, focal) = 2 * rad2deg(atan(resolution / (2 * focal)))

"""
The first of `candidates` present in `dir`, erroring out when none is.
"""
function rc_source_file(dir::String, candidates, what::String)
    for candidate in candidates
        path = joinpath(dir, candidate)
        isfile(path) && return path
    end
    error("RealityCapture export is missing its $what: none of " *
        join(("`$c`" for c in candidates), ", ") * " in `$dir`.")
end

"""
    rc_convert(source_dir; kwargs...)

Convert a RealityCapture / RealityScan export into the COLMAP layout
`ColmapDataset` reads: undistorted `images/` & a
`sparse/0/{cameras,images,points3D}.bin` model.

`source_dir` is expected to hold

    input/             the images, as exported
    bundler.out        the Bundler v0.3 export: poses, points & SfM tracks
    images.txt         the Bundler image list (`list.txt` also accepted)
    camera-params.csv  the camera & lens table (`cameras.csv` also accepted)

The Bundler export is required: it is the only part of an RC export carrying SfM
tracks, and the depth anchors are fitted on the points each view actually saw.
Either of RC's two Bundler options works — the axis convention is detected, not
assumed (see [`bundler_flip`](@ref)).

`points.ply` is optional & folded in only when it holds a *different* cloud than
the `.out`, which is RC's dense export: a far better initialization than the
sparse Bundler cloud, carrying no tracks of its own.

- `output_dir`: where the conversion is written, `source_dir` by default.
- `resize`: also write 1/2, 1/4 & 1/8 scale image directories.
- `max_points`: randomly subsample each init cloud down to this many points.
    `0` keeps them whole. RC clouds carry far-field background the cameras
    never get near, so capping is sometimes worth it.
- `focal_tolerance`: drop views whose focal deviates from the median by more
    than this fraction (see `RC_FOCAL_TOLERANCE`).
"""
function rc_convert(source_dir::String;
    output_dir::String = source_dir, resize::Bool = false,
    max_points::Int = 0, focal_tolerance::Float64 = RC_FOCAL_TOLERANCE,
)
    images_dir = joinpath(source_dir, RC_IMAGES_DIR)
    isdir(images_dir) || error("RealityCapture export is missing `$images_dir`.")
    bundler_file = joinpath(source_dir, RC_BUNDLER_FILE)
    isfile(bundler_file) || error("RealityCapture export is missing `$bundler_file`.")
    params_file = rc_source_file(source_dir, RC_PARAMS_FILES, "camera table")
    list_file = rc_source_file(source_dir, RC_BUNDLER_LISTS, "Bundler image list")

    params = read_rc_params(params_file)
    @info "Read `$(length(params))` camera(s) from `$params_file`."
    by_name = Dict(view.name => view for view in params)

    bundler = read_bundler(bundler_file)
    names = read_bundler_list(list_file)
    length(names) == length(bundler.cameras) || error(
        "`$list_file` names $(length(names)) image(s) but `$bundler_file` has " *
        "$(length(bundler.cameras)) camera(s).")
    @info "Read `$(length(bundler.cameras))` camera(s) & " *
        "`$(length(bundler.points))` point(s) from `$bundler_file`."

    # `kept` holds the Bundler camera indices that make it into the model, in
    # order. The tracks are remapped through it, so dropping a view here cannot
    # leave a stale camera index behind.
    kept = collect(eachindex(bundler.cameras))
    function drop!(predicate, reason::String)
        dropped = filter(predicate, kept)
        isempty(dropped) && return
        @warn "Dropped $(length(dropped)) view(s), $reason" * (
            length(dropped) ≤ 8 ?
                ": " * join((names[i] for i in dropped), ", ") * "." : ".")
        filter!(!predicate, kept)
        return
    end

    drop!(i -> bundler.cameras[i].focal ≤ 0, "unregistered in `$bundler_file`")
    drop!(i -> !haskey(by_name, names[i]), "no row in `$params_file`")
    drop!(i -> !isfile(joinpath(images_dir, names[i])), "no image in `$images_dir`")
    isempty(kept) && error("No view survived: nothing to convert.")

    # A zoomed frame warped into the shared pinhole would be a small image in a
    # black frame, and the border would be trained against.
    median_focal = median(by_name[names[i]].f_35mm for i in kept)
    drop!(i -> abs(by_name[names[i]].f_35mm - median_focal) > focal_tolerance * median_focal,
        "focal deviates from the median $(round(median_focal; digits=2))mm by " *
        "more than $(round(focal_tolerance * 100; digits=1))%")
    isempty(kept) && error("No view survived the focal filter.")

    views = [by_name[names[i]] for i in kept]
    registered = Set(names)
    n_unregistered = count(readdir(images_dir)) do file
        isfile(joinpath(images_dir, file)) && !(file in registered)
    end
    n_unregistered > 0 && @info(
        "`$images_dir` holds $n_unregistered file(s) with no camera — " *
        "unregistered frames & sidecars are ignored.")

    height, width = size(load(joinpath(images_dir, first(views).name)))
    @info "Image resolution: ($width x $height) (width x height)."

    focal = maximum(v -> rc_required_focal(v, width, height), views)
    @info "Undistorting into a shared pinhole: " *
        "focal $(round(focal; digits=2))px, " *
        "fov $(round(focal2fov(width, focal); digits=2))° x " *
        "$(round(focal2fov(height, focal); digits=2))°."

    flip = bundler_flip(bundler, width, height)

    out_images_dir = joinpath(output_dir, "images")
    mkpath(out_images_dir)
    scales = resize ? (2, 4, 8) : ()
    for scale in scales
        mkpath(joinpath(output_dir, "images_$scale"))
    end

    progress = ProgressMeter.Progress(length(views); desc="Undistorting: ")
    Threads.@threads for view in views
        image = load(joinpath(images_dir, view.name))
        # One camera covers every view, so a frame of another size would be
        # trained against intrinsics that are not its own.
        size(image) == (height, width) || error(
            "`$(view.name)` is $(size(image, 2))x$(size(image, 1)), but the model " *
            "assumes $(width)x$(height): every view has to share one resolution.")

        undistorted = rc_undistort(image, view, focal)
        save(joinpath(out_images_dir, view.name), undistorted)
        for scale in scales
            save(
                joinpath(output_dir, "images_$scale", view.name),
                imresize(undistorted, (height ÷ scale, width ÷ scale)))
        end
        ProgressMeter.next!(progress)
    end
    ProgressMeter.finish!(progress)

    # Poses first: the tracks below address images by position in `kept`.
    images = map(enumerate(kept)) do (id, i)
        camera = bundler.cameras[i]
        R, t = flip * camera.R, flip * camera.t
        ColmapImage(names[i],
            SVector{4, Float64}(Rotations.params(QuatRotation(R))),
            SVector{3, Float64}(t),
            Tuple{Float64, Float64, UInt64}[])
    end

    # Bundler camera index -> output image id, `0` for a dropped view.
    image_id = zeros(Int, length(bundler.cameras))
    for (id, i) in enumerate(kept)
        image_id[i] = id
    end

    selected = eachindex(bundler.points)
    if max_points > 0 && length(bundler.points) > max_points
        selected = sort!(randperm(length(bundler.points))[1:max_points])
        @info "Subsampled the Bundler cloud to `$max_points` points."
    end

    positions = SVector{3, Float64}[]
    colors = SVector{3, UInt8}[]
    tracks = Vector{Tuple{UInt32, UInt32}}[]
    for i in selected
        point = bundler.points[i]
        push!(positions, point.position)
        push!(colors, point.color)

        point_id = UInt64(length(positions))
        track = Tuple{UInt32, UInt32}[]
        for observation in point.views
            id = get(image_id, observation.camera + 1, 0)
            id == 0 && continue
            # Bundler measures from the image center with `+y` up.
            # NOTE: in the *distorted* frame — `rc_undistort` moves the pixel, and
            # inverting that per observation buys nothing, as `dataset.jl` reads
            # the point ids & never the coordinates.
            observations = images[id].observations
            push!(observations,
                (observation.x + width / 2, height / 2 - observation.y, point_id))
            push!(track, (UInt32(id), UInt32(length(observations) - 1)))
        end
        push!(tracks, track)
    end
    n_bundler_points = length(positions)

    # A denser `points.ply` initializes far better than the sparse Bundler cloud.
    # That it carries no tracks is harmless: `collect_anchor_samples` only ever
    # samples the columns a view's track names, so the extra points feed
    # initialization & stay out of the depth-anchor fit.
    points_file = joinpath(source_dir, RC_POINTS_FILE)
    if isfile(points_file)
        ply_points, ply_colors = read_rc_points(points_file)
        @info "Read `$(size(ply_points, 2))` point(s) from `$points_file`."
        if size(ply_points, 2) == length(bundler.points)
            @info "`$points_file` is the same cloud as `$bundler_file` — ignoring it."
        elseif !rc_enu_matches_bundler(
            (bundler.cameras[i] for i in kept), (by_name[names[i]].position for i in kept))
            @warn "Ignoring `$points_file`: RealityCapture's ENU frame does not line " *
                "up with the Bundler frame on this export, so the two clouds cannot " *
                "be mixed."
        else
            keep = axes(ply_points, 2)
            if max_points > 0 && length(keep) > max_points
                keep = sort!(randperm(size(ply_points, 2))[1:max_points])
                @info "Subsampled `$points_file` to `$max_points` points."
            end
            for i in keep
                push!(positions, RC_ENU_TO_BUNDLER * SVector{3, Float64}(
                    ply_points[1, i], ply_points[2, i], ply_points[3, i]))
                push!(colors, SVector{3, UInt8}(
                    ply_colors[1, i], ply_colors[2, i], ply_colors[3, i]))
                push!(tracks, Tuple{UInt32, UInt32}[])
            end
            @info "Added `$(length(keep))` untracked init point(s) from `$points_file`."
        end
    end

    sparse_dir = joinpath(output_dir, "sparse", "0")
    mkpath(sparse_dir)
    write_colmap_cameras(joinpath(sparse_dir, "cameras.bin"), width, height, focal)
    write_colmap_images(joinpath(sparse_dir, "images.bin"), images)
    write_colmap_points(joinpath(sparse_dir, "points3D.bin"), positions, colors, tracks)

    observations = sum(length, tracks)
    @info "Wrote `$(length(images))` image(s) & `$(length(positions))` point(s) " *
        "($n_bundler_points tracked, $observations observation(s), " *
        "$(round(observations / length(images); digits=1)) per view) to `$output_dir`."
    return output_dir
end
