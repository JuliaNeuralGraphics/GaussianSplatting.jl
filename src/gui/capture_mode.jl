# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
Base.@kwdef mutable struct CaptureMode
    camera_path::CameraPath = CameraPath()
    writer::Union{VideoIO.VideoWriter, Nothing} = nothing
    is_rendering::Bool = false

    # UI stuff.
    save_frames::Ref{Bool} = Ref(false)
    steps_ref::Ref{Int32} = Ref{Int32}(24)
    framerate_ref::Ref{Int32} = Ref{Int32}(24)
    save_dir::Vector{UInt8} = Vector{UInt8}("./" * "\0"^254)
end

function advance!(v::CaptureMode, gui)
    if is_done(v.camera_path)
        v.is_rendering = false
        v.writer ≢ nothing && close_video_out!(v.writer)
    else
        k = current_pose(v.camera_path)
        set_c2w!(gui.camera, NU.get_rotation(k), k.t)
    end
    advance!(v.camera_path, get_time_step(v.camera_path, v.steps_ref[]))
end

function close_video!(v::CaptureMode)
    if v.writer ≢ nothing
        close_video_out!(v.writer)
        v.writer = nothing
    end
end

function reset!(v::CaptureMode)
    close_video!(v)
    empty!(v.camera_path)
    v.save_dir = Vector{UInt8}("./" * "\0"^254)
end

function handle_ui!(capture_mode::CaptureMode; gui)
    if CImGui.Begin("Capture Mode")
        CImGui.TextWrapped("Capturing with the same render settings as in the main screen.")
        CImGui.TextWrapped("Window is not resizable during this stage.")
        CImGui.TextWrapped("Press V to add camera position (at least 2 are required).")
        CImGui.Text("N Keyframes: $(length(capture_mode.camera_path))")

        if capture_mode.is_rendering
            CImGui.TextWrapped("Capturing. Please wait...")

            n_steps = capture_mode.steps_ref[] * (length(capture_mode.camera_path) - 1)
            current_step = capture_mode.camera_path.current_step

            CImGui.PushStyleColor(CImGui.ImGuiCol_PlotHistogram, CImGui.HSV(0.61f0, 1.0f0, 1f0))
            CImGui.ProgressBar(current_step / (n_steps + 1), CImGui.ImVec2(-1f0, 0f0),
                "$current_step / $(n_steps + 1)")
            CImGui.PopStyleColor()

            if CImGui.Button("Cancel", CImGui.ImVec2(-1, 0))
                capture_mode.is_rendering = false
                close_video!(capture_mode)
            end
        else
            if CImGui.CollapsingHeader("Video Settings", CIM_HEADER)
                CImGui.PushItemWidth(-100)
                CImGui.SliderInt("Lerp Steps", capture_mode.steps_ref, 1, 60, "%d / 60")
                CImGui.PushItemWidth(-100)
                CImGui.SliderInt("Frame rate", capture_mode.framerate_ref, 1, 60, "%d")

                CImGui.PushItemWidth(-100)
                CImGui.InputText("Save Directory", pointer(capture_mode.save_dir),
                    length(capture_mode.save_dir))

                CImGui.PushItemWidth(-100)
                CImGui.Checkbox("Save frames", capture_mode.save_frames)
            end

            CImGui.BeginTable("##capture-buttons-table", 3)
            CImGui.TableNextRow()
            CImGui.TableNextColumn()

            can_capture = length(capture_mode.camera_path) ≥ 2
            can_capture || disabled_begin()
            if CImGui.Button("Capture", CImGui.ImVec2(-1, 0))
                reset_time!(capture_mode.camera_path)
                close_video!(capture_mode)

                # Create directories for video & images.
                save_dir = unsafe_string(pointer(capture_mode.save_dir))
                isdir(save_dir) || mkdir(save_dir)
                if capture_mode.save_frames[]
                    images_dir = joinpath(save_dir, "images")
                    isdir(images_dir) || mkdir(images_dir)
                end

                # Open video writer stream.
                video_file = joinpath(save_dir, "out.mp4")
                res = resolution(gui.camera)
                capture_mode.writer = open_video_out(
                    video_file, zeros(RGB{N0f8}, res.height, res.width);
                    framerate=capture_mode.framerate_ref[],
                    target_pix_fmt=VideoIO.AV_PIX_FMT_YUV420P)

                capture_mode.is_rendering = true
            end
            can_capture || disabled_end()

            CImGui.TableNextColumn()
            if CImGui.Button("Go Back", CImGui.ImVec2(-1, 0))
                gui.screen = MainScreen
                capture_mode.is_rendering = false
                close_video!(capture_mode)
                NGL.set_resizable_window!(gui.context, true)
            end

            CImGui.TableNextColumn()
            red_button_begin()
            if CImGui.Button("Clear Path", CImGui.ImVec2(-1, 0))
                empty!(capture_mode.camera_path)
                gui.render_state.need_render = true
            end
            red_button_end()
            CImGui.EndTable()
        end
    end
    CImGui.End()
end

function loop!(capture_mode::CaptureMode; gui)
    frame_time = update_time!(gui.render_state)

    NGL.imgui_begin()
    handle_ui!(capture_mode; gui)

    if !capture_mode.is_rendering && !is_mouse_in_ui()
        controller_id = gui.ui_state.controller_mode[]

        gui.render_state.need_render |= handle_keyboard!(
            gui.control_settings, gui.camera; frame_time, controller_id)
        gui.render_state.need_render |= handle_mouse!(
            gui.control_settings, gui.camera; controller_id)

        if NGL.is_key_pressed(iglib.ImGuiKey_V; repeat=false)
            push!(capture_mode.camera_path, deepcopy(gui.camera))
            gui.render_state.need_render = true
        end
    end

    NGL.clear()
    NGL.set_clear_color(0.2, 0.2, 0.2, 1.0)

    if capture_mode.is_rendering
        # Initial advance.
        if capture_mode.camera_path.current_step == 0
            advance!(capture_mode, gui)
            gui.render_state.need_render = true
        end
    end

    render!(gui)
    NGL.draw(gui.render_state.surface)
    NGL.clear(NGL.GL_DEPTH_BUFFER_BIT)

    if !capture_mode.is_rendering
        P = NGL.perspective(gui.camera)
        L = NGL.look_at(gui.camera)
        NGL.draw(capture_mode.camera_path, P, L; frustum=gui.frustum)
    end

    NGL.imgui_end()
    GLFW.SwapBuffers(gui.context.window)
    GLFW.PollEvents()

    # Save rendered frame.
    if capture_mode.is_rendering
        mode = gui.ui_state.selected_mode[]

        frame = if mode == 0 # Render color.
            to_image(gui.rasterizer)
        elseif mode == 1 # Render depth.
            to_depth(gui.rasterizer)
        end .|> RGB{N0f8}

        save_dir = unsafe_string(pointer(capture_mode.save_dir))
        if capture_mode.save_frames[]
            images_dir = joinpath(save_dir, "images")
            image_file = "gsp-$(capture_mode.camera_path.current_step).png"
            save(joinpath(images_dir, image_file), frame)
        end
        write(capture_mode.writer, frame)

        advance!(capture_mode, gui)
        capture_mode.is_rendering && (gui.render_state.need_render = true;)
    end
    return
end
