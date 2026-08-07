# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
Base.@kwdef mutable struct CameraPath
    keyframes::Vector{NU.CameraKeyframe} = []
    current_time::Float32 = 0f0
    current_step::Int64 = 0

    frustums::Vector{FrustumPose} = []
    gui_lines::Vector{NGL.Line} = []
    line_program::NGL.ShaderProgram = NGL.get_program(NGL.Line)
end

Base.push!(p::CameraPath, camera::Camera) =
    push!(p, NU.CameraKeyframe(camera), camera_tan_half(camera))

# A keyframe loaded from a file has no `Camera` behind it, so the size of the
# frustum drawn for it is given explicitly (see `load_camera_path!`).
function Base.push!(
    p::CameraPath, keyframe::NU.CameraKeyframe, tan_half::SVector{2, Float32},
)
    push!(p.keyframes, keyframe)
    push!(p.frustums, FrustumPose(
        NU.get_rotation(keyframe), keyframe.t, tan_half))
    length(p.keyframes) == 1 && return

    from = p.keyframes[end - 1].t
    push!(p.gui_lines, NGL.Line(from, keyframe.t; program=p.line_program))
    return
end

function Base.getindex(p::CameraPath, i::Integer)
    isempty(p.keyframes) &&
        error("Tried accessing an empty CameraPath at `$i` index.")
    p.keyframes[clamp(i, 1, length(p.keyframes))]
end

Base.length(p::CameraPath) = length(p.keyframes)

Base.isempty(p::CameraPath) = isempty(p.keyframes)

function get_time_step(p::CameraPath, n_steps::Integer)
    1f0 / (n_steps * (length(p) - 1))
end

function advance!(p::CameraPath, δ::Float32)
    p.current_time += δ
    p.current_step += 1
end

is_done(p::CameraPath) = p.current_time ≥ 1f0

function reset_time!(p::CameraPath)
    p.current_time = 0f0
    p.current_step = 0
end

function Base.empty!(p::CameraPath)
    p.current_time = 0f0
    p.current_step = 0

    NGL.delete!.(p.gui_lines; with_program=false)
    empty!(p.keyframes)
    empty!(p.frustums)
    empty!(p.gui_lines)
end

"""
# Arguments:

- `t::Float32`: Time value in `[0, 1]` range. Goes through all keyframes.
"""
function current_pose(p::CameraPath)
    t = p.current_time
    t = t * (length(p) - 1)
    idx = floor(Int, t) + 1
    NU.spline(t - floor(t), p[idx - 1], p[idx], p[idx + 1], p[idx + 2])
end

function NGL.draw(p::CameraPath, P, L;
    renderer::FrustumRenderer, scale::Float32,
)
    isempty(p.gui_lines) && return
    for l in p.gui_lines
        NGL.draw(l, P, L)
    end

    # Draw keyframe poses, in the color of the path lines they connect.
    draw_wireframes(renderer, p.frustums, P, L;
        scale, color=SVector{4, Float32}(0.8f0, 0.5f0, 0.1f0, 1f0))
end
