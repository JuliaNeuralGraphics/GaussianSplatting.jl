Base.@kwdef mutable struct RenderState
    surface::NeuralGraphicsGL.RenderSurface
    need_render::Bool = true # `true` to trigger first frame rendering
    last_frame_time::Float64 = time()
end

function update_time!(s::RenderState)
    now = time()
    frame_time = now - s.last_frame_time
    s.last_frame_time = now
    return frame_time
end

struct ControlSettings
    camera_velocity::Float32
    rotation_sensitivity::Float32
    up_vec::SVector{3, Float32}

    function ControlSettings(;
        camera_velocity::Float32 = 4f0,
        rotation_sensitivity::Float32 = 0.005f0,
    )
        up_vec = SVector{3, Float32}(0f0, -1f0, 0f0)
        new(camera_velocity, -abs(rotation_sensitivity), up_vec)
    end
end

function handle_keyboard!(
    control_settings::ControlSettings, camera::Camera; frame_time::Real,
)::Bool
    need_render = false
    translate_vec = zeros(MVector{3, Float32})

    if NeuralGraphicsGL.is_key_down('A')
        translate_vec[1] -= 1f0
        need_render = true
    elseif NeuralGraphicsGL.is_key_down('D')
        translate_vec[1] += 1f0
        need_render = true
    end
    if NeuralGraphicsGL.is_key_down('W')
        translate_vec[3] += 1f0
        need_render = true
    elseif NeuralGraphicsGL.is_key_down('S')
        translate_vec[3] -= 1f0
        need_render = true
    end
    if NeuralGraphicsGL.is_key_down('E')
        translate_vec[2] -= 1f0
        need_render = true
    elseif NeuralGraphicsGL.is_key_down('Q')
        translate_vec[2] += 1f0
        need_render = true
    end

    if need_render
        translate_vec .*= control_settings.camera_velocity * frame_time
        shift!(camera, translate_vec)
    end
    return need_render
end

function handle_mouse!(control_settings::ControlSettings, camera::Camera)
    io = CImGui.GetIO()
    do_handle_mouse =
        CImGui.IsMousePosValid() &&
        unsafe_load(io.WantCaptureMouse) == false &&
        CImGui.IsMouseDown(0)

    need_render = false
    if do_handle_mouse
        mouse_δ = NeuralGraphicsGL.get_mouse_delta()
        δx = mouse_δ.x * control_settings.rotation_sensitivity
        δy = mouse_δ.y * control_settings.rotation_sensitivity
        if NeuralGraphicsGL.is_key_down('R') # roll
            R = AngleAxis(δy, view_dir(camera)...)
        else
            R = AngleAxis(δx, control_settings.up_vec...) *
                AngleAxis(δy, view_side(camera)...)
        end

        rotate!(camera, R)
        need_render = true
    end
    need_render
end
