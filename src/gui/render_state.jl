# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
Base.@kwdef mutable struct RenderState
    surface::NeuralGraphicsGL.RenderSurface
    need_render::Bool = true # `true` to trigger first frame rendering
    last_frame_time::Float64 = time()
end

function update_time!(s::RenderState)::Float64
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

    io = CImGui.GetIO()
    do_handle_mouse =
        CImGui.IsMousePosValid() &&
        unsafe_load(io.WantCaptureMouse) == false &&
        CImGui.IsMouseDown(0)
    do_handle_mouse || return need_render

    mouse_δ = NeuralGraphicsGL.get_mouse_delta()
    δx = mouse_δ.x * control_settings.rotation_sensitivity
    δy = mouse_δ.y * control_settings.rotation_sensitivity
    δx ≈ 0f0 && δy ≈ 0f0 && return need_render

    if NeuralGraphicsGL.is_key_down(iglib.ImGuiKey_R) # roll
        R = AngleAxis(δy, view_dir(camera)...)
    else
        R = AngleAxis(δx, -view_up(camera)...) *
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

    io = CImGui.GetIO()
    lmouse = CImGui.IsMouseDown(0)
    rmouse = CImGui.IsMouseDown(1)
    do_handle_mouse =
        CImGui.IsMousePosValid() &&
        unsafe_load(io.WantCaptureMouse) == false && (lmouse || rmouse)
    do_handle_mouse || return need_render

    mouse_δ = NeuralGraphicsGL.get_mouse_delta()
    δx = mouse_δ.x * control_settings.rotation_sensitivity
    δy = mouse_δ.y * control_settings.rotation_sensitivity
    δx ≈ 0f0 && δy ≈ 0f0 && return need_render

    if lmouse
        # Translate camera.
        translate_vec = if NeuralGraphicsGL.is_key_down(iglib.ImGuiKey_F)
            SVector{3, Float32}(δx, δy, 0)
        else
            SVector{3, Float32}(δx, 0f0, -δy * 10f0)
        end
        shift!(camera, translate_vec)

        # TODO adjust distance to target
        control_settings.orbiting_target = view_pos(camera) .+ 10f0 .* view_dir(camera)
    elseif rmouse
        sensitivity = 5f0
        translate_vec = SVector{3, Float32}(δx * sensitivity, δy * sensitivity, 0f0)
        shift!(camera, translate_vec)

        target = control_settings.orbiting_target
        LT = NGL.look_at(
            view_pos(camera), target, view_up(camera);
            left_handed=false)

        new_c2w = inv(LT)
        set_c2w!(camera, new_c2w)
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
