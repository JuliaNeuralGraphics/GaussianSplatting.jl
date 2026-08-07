# TOML (de)serialization for `CameraPath`.
#
# Only the keyframes are stored: everything else a path carries (its spline
# time, the lines & frustums drawn for it) is derived from them.

"""
    save_camera_path(path::AbstractString, p::CameraPath)

Write the keyframes of `p` as TOML, in path order.

Each keyframe is the camera-to-world pose the path interpolates:
its `position` (camera center) & its `rotation`, as a `[w, x, y, z]` quaternion.
"""
function save_camera_path(path::AbstractString, p::CameraPath)
    isempty(p) && error("Refusing to write an empty camera path.")

    keyframes = [
        Dict{String, Any}(
            "position" => [to_toml(v) for v in k.t],
            "rotation" => [to_toml(v) for v in (k.q.s, k.q.v1, k.q.v2, k.q.v3)])
        for k in p.keyframes]

    open(path, "w") do io
        println(io, "# GaussianSplatting.jl camera path.")
        println(io, "# Load with `Load Path...` in the GUI's `Capture` tab.")
        TOML.print(io, Dict{String, Any}("keyframe" => keyframes); sorted=true)
    end
    return path
end

"""
    load_camera_path(path::AbstractString)::Vector{NU.CameraKeyframe}

Read the keyframes written by [`save_camera_path`](@ref).

Throws if the file is not a camera path, if a keyframe is malformed, or if
there are fewer than the 2 keyframes a capture needs.
"""
function load_camera_path(path::AbstractString)
    entries = TOML.parsefile(path)

    raw = get(entries, "keyframe", nothing)
    raw isa AbstractVector ||
        error("Camera path file has no `[[keyframe]]` entries.")
    length(raw) ≥ 2 ||
        error("A camera path needs at least 2 keyframes, got $(length(raw)).")

    return NU.CameraKeyframe[
        keyframe_from_toml(entry, i) for (i, entry) in enumerate(raw)]
end

function keyframe_from_toml(entry, i::Integer)
    entry isa AbstractDict || error("Keyframe $i is not a table.")

    t = toml_vec(entry, "position", 3, i)
    q = toml_vec(entry, "rotation", 4, i)

    # The spline & `NU.get_rotation` both assume a unit quaternion; a
    # hand-edited file need not have one, so normalize rather than reject.
    n = sqrt(sum(abs2, q))
    n > 0f0 || error("Keyframe $i: `rotation` is all zeros.")
    q = q ./ n

    return NU.CameraKeyframe(
        QuaternionF32(q[1], q[2], q[3], q[4]), SVector{3, Float32}(t))
end

function toml_vec(entry::AbstractDict, key::String, n::Integer, i::Integer)
    value = get(entry, key, nothing)
    value isa AbstractVector || error("Keyframe $i is missing `$key`.")
    length(value) == n ||
        error("Keyframe $i: `$key` needs $n numbers, got $(length(value)).")
    all(v -> v isa Real, value) ||
        error("Keyframe $i: `$key` must be numbers, got `$value`.")
    return Float32.(value)
end

"""
    load_camera_path!(p::CameraPath, path::AbstractString, tan_half)

Replace the keyframes of `p` with the ones in the file at `path`.

`tan_half` sizes the frustums drawn for the loaded keyframes - the file does
not describe a lens, so this is the viewing camera's (see `camera_tan_half`).

Creates & deletes OpenGL objects, so it must run on the render thread.
`p` is left untouched if the file does not parse.
"""
function load_camera_path!(
    p::CameraPath, path::AbstractString, tan_half::SVector{2, Float32},
)
    keyframes = load_camera_path(path)
    empty!(p)
    for keyframe in keyframes
        push!(p, keyframe, tan_half)
    end
    return p
end
