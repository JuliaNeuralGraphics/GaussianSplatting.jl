# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
Base.@kwdef mutable struct RenderState
    surface::NeuralGraphicsGL.RenderSurface
    framebuffer::NeuralGraphicsGL.Framebuffer
    need_render::Bool = true # `true` to trigger first frame rendering
    last_frame_time::Float64 = time()
end

function update_time!(s::RenderState)
    now = time()
    frame_time = now - s.last_frame_time
    s.last_frame_time = now
    return frame_time
end

mutable struct ControlSettings
    camera_velocity::Float32
    rotation_sensitivity::Float32
    up_vec::SVector{3, Float32}
    orbiting_target::SVector{3, Float32}

    function ControlSettings(;
        camera_velocity::Float32 = 8f0,
        rotation_sensitivity::Float32 = 0.005f0,
    )
        up_vec = SVector{3, Float32}(0f0, -1f0, 0f0)
        new(camera_velocity, -abs(rotation_sensitivity), up_vec,
            SVector{3, Float32}(0f0, 0f0, 0f0))
    end
end

"""
Estimate the scene's up direction as the average of the dataset
cameras' up axes: photos are mostly taken with a roughly level camera,
while the COLMAP world orientation is arbitrary. Yawing around this
vector instead of the world `-Y` keeps the horizon level.
Negated to match the `up_vec` convention (`-view_up` of an identity camera).
"""
function estimate_up_vec(cameras::Vector{Camera})
    return -normalize(sum(view_up, cameras))
end

"""
Look from `pos` towards `target` with zero roll w.r.t. `up_vec`:
rebuild the orientation frame in `c2w` convention
(columns: side, up, dir; `up_vec = -view_up`). See also `estimate_up_vec`.
"""
function look_at!(
    c::Camera, pos::SVector{3, Float32}, target::SVector{3, Float32},
    up_vec::SVector{3, Float32},
)
    dir = normalize(target - pos)
    side = dir × up_vec
    n = norm(side)
    n ≈ 0f0 && return # Looking straight along `up_vec`: side is ambiguous.
    side = side ./ n

    up = dir × side
    R = hcat(side, up, dir)
    set_c2w!(c, R, pos)
    return
end

# Remove camera roll w.r.t. `up_vec`, keeping the position & view direction.
function level_horizon!(c::Camera, up_vec::SVector{3, Float32})
    look_at!(c, view_pos(c), view_pos(c) + view_dir(c), up_vec)
end

function fpv_keyboard_controller(
    control_settings::ControlSettings, camera::Camera; frame_time::Real,
)
    need_render = false
    translate_vec = zeros(MVector{3, Float32})

    if NeuralGraphicsGL.is_key_down(iglib.ImGuiKey_A)
        translate_vec[1] -= 1f0
        need_render = true
    elseif NeuralGraphicsGL.is_key_down(iglib.ImGuiKey_D)
        translate_vec[1] += 1f0
        need_render = true
    end
    if NeuralGraphicsGL.is_key_down(iglib.ImGuiKey_W)
        translate_vec[3] += 1f0
        need_render = true
    elseif NeuralGraphicsGL.is_key_down(iglib.ImGuiKey_S)
        translate_vec[3] -= 1f0
        need_render = true
    end
    if NeuralGraphicsGL.is_key_down(iglib.ImGuiKey_E)
        translate_vec[2] -= 1f0
        need_render = true
    elseif NeuralGraphicsGL.is_key_down(iglib.ImGuiKey_Q)
        translate_vec[2] += 1f0
        need_render = true
    end

    if need_render
        translate_vec .*= control_settings.camera_velocity * frame_time
        shift!(camera, translate_vec)
    end

    return need_render
end

function orbiting_keyboard_controller(
    control_settings::ControlSettings, camera::Camera; frame_time::Real,
)
    need_render = false
    return need_render
end

function handle_keyboard!(
    control_settings::ControlSettings, camera::Camera;
    frame_time::Real, controller_id::Integer,
)::Bool
    need_render = if controller_id == 0
        fpv_keyboard_controller(control_settings, camera; frame_time)
    else
        orbiting_keyboard_controller(control_settings, camera; frame_time)
    end
    return need_render
end

function fpv_mouse_controller(
    control_settings::ControlSettings, camera::Camera,
)
    need_render = false

    # Callers gate on the scene view being hovered.
    do_handle_mouse = CImGui.IsMousePosValid() && CImGui.IsMouseDown(0)
    do_handle_mouse || return need_render

    mouse_δ = NeuralGraphicsGL.get_mouse_delta()
    δx = mouse_δ.x * control_settings.rotation_sensitivity
    δy = mouse_δ.y * control_settings.rotation_sensitivity
    δx ≈ 0f0 && δy ≈ 0f0 && return need_render

    if NeuralGraphicsGL.is_key_down(iglib.ImGuiKey_R) # roll
        R = AngleAxis(δy, view_dir(camera)...)
    else
        # Clamp pitch to ±89° from the horizon: crossing the pole
        # would flip the yaw direction.
        up_vec = control_settings.up_vec
        pitch = asin(clamp(view_dir(camera) ⋅ up_vec, -1f0, 1f0))
        pitch_limit = deg2rad(89f0)
        δy = clamp(pitch + δy, -pitch_limit, pitch_limit) - pitch

        R = AngleAxis(δx, up_vec...) *
            AngleAxis(δy, view_side(camera)...)
    end

    rotate!(camera, R)
    need_render = true
    return need_render
end

function orbiting_mouse_controller(
    control_settings::ControlSettings, camera::Camera;
)
    need_render = false
    up_vec = control_settings.up_vec
    target = control_settings.orbiting_target

    # Zoom towards the target with the mouse wheel.
    wheel = unsafe_load(CImGui.GetIO().MouseWheel)
    if !(wheel ≈ 0f0)
        r = norm(view_pos(camera) - target)
        r_new = max(1f-2, r * 0.9f0^wheel)
        shift!(camera, SVector{3, Float32}(0f0, 0f0, r - r_new))
        need_render = true
    end

    lmouse = CImGui.IsMouseDown(0)
    rmouse = CImGui.IsMouseDown(1)
    # Callers gate on the scene view being hovered.
    do_handle_mouse = CImGui.IsMousePosValid() && (lmouse || rmouse)
    do_handle_mouse || return need_render

    mouse_δ = NeuralGraphicsGL.get_mouse_delta()
    δx = mouse_δ.x * control_settings.rotation_sensitivity
    δy = mouse_δ.y * control_settings.rotation_sensitivity
    δx ≈ 0f0 && δy ≈ 0f0 && return need_render

    if lmouse
        # Pan: translate camera & target in the view plane,
        # scaled by the distance to the target.
        r = norm(view_pos(camera) - target)
        translate_vec = SVector{3, Float32}(δx * r, δy * r, 0f0)
        world_δ = SMatrix{3, 3, Float32}(@view(camera.c2w[1:3, 1:3])) * translate_vec
        shift!(camera, translate_vec)
        control_settings.orbiting_target = target .+ world_δ
    elseif rmouse
        # Orbit around the target: yaw about the scene up (no roll),
        # pitch about the side axis, clamped to not cross the pole.
        offset = view_pos(camera) - target
        r = norm(offset)
        r ≈ 0f0 && return need_render

        dir = -offset ./ r # Viewing direction, towards the target.
        side = dir × up_vec
        n = norm(side)
        n ≈ 0f0 && return need_render
        side = side ./ n

        pitch = asin(clamp(dir ⋅ up_vec, -1f0, 1f0))
        pitch_limit = deg2rad(89f0)
        δy = clamp(pitch + δy, -pitch_limit, pitch_limit) - pitch

        R = AngleAxis(δx, up_vec...) * AngleAxis(δy, side...)
        look_at!(camera, target .+ R * offset, target, up_vec)
    end

    need_render = true
    return need_render
end

function handle_mouse!(
    control_settings::ControlSettings, camera::Camera; controller_id::Integer,
)
    need_render = if controller_id == 0
        fpv_mouse_controller(control_settings, camera)
    else
        orbiting_mouse_controller(control_settings, camera)
    end
    return need_render
end
