"""
Video capture along a user-defined camera path.

Frames come from the background worker, like the interactive view:
the UI points the camera at the current path step, publishes it &
waits for the worker to hand back the frame rendered from exactly that snapshot (see `poll_capture!`).
"""
Base.@kwdef mutable struct CaptureMode
    camera_path::CameraPath = CameraPath()
    writer::Union{VideoIO.VideoWriter, Nothing} = nothing
    is_rendering::Bool = false

    # In-flight request: the snapshot whose frame goes into the video next &
    # the frame size the video stream was opened with.
    pending_version::UInt64 = 0
    width::Int = 0
    height::Int = 0
    # Total frames in the camera path & how many are written already.
    total_frames::Int = 0
    frame_index::Int = 0

    # UI stuff.
    save_frames::Ref{Bool} = Ref(false)
    steps_ref::Ref{Int32} = Ref{Int32}(24)
    framerate_ref::Ref{Int32} = Ref{Int32}(24)
    save_dir::Vector{UInt8} = Vector{UInt8}("./" * "\0"^254)
end

# Frames the current path produces: `steps_ref` per keyframe segment.
path_frames(c::CaptureMode) = Int(c.steps_ref[]) * (length(c.camera_path) - 1)

function close_video!(c::CaptureMode)
    if c.writer ≢ nothing
        close_video_out!(c.writer)
        c.writer = nothing
    end
end

# Stop a capture in progress, finalizing whatever was written so far.
function stop_capture!(c::CaptureMode)
    c.is_rendering = false
    close_video!(c)
    return
end

"""
Drop the camera path & any capture in progress.
Deletes OpenGL objects (the path lines), so it must run on the render thread;
called whenever the scene the path was drawn for goes away.
"""
function reset!(c::CaptureMode)
    stop_capture!(c)
    empty!(c.camera_path)
    return
end

save_directory(c::CaptureMode) = unsafe_string(pointer(c.save_dir))

# Replace the save path, keeping the buffer NUL-padded so that the
# `InputText` field stays editable afterwards.
function set_save_directory!(c::CaptureMode, dir::AbstractString)
    c.save_dir = Vector{UInt8}(dir * "\0"^256)
    return
end

function start_capture!(c::CaptureMode, gui)
    reset_time!(c.camera_path)
    close_video!(c)

    # Create directories for video & images.
    save_dir = save_directory(c)
    mkpath(save_dir)
    c.save_frames[] && mkpath(joinpath(save_dir, "images"))

    (; width, height) = resolution(gui.camera)
    c.width, c.height = width, height
    c.total_frames = path_frames(c)
    c.frame_index = 0

    # Open video writer stream.
    c.writer = open_video_out(
        joinpath(save_dir, "out.mp4"), zeros(RGB{N0f8}, height, width);
        framerate=c.framerate_ref[],
        target_pix_fmt=VideoIO.AV_PIX_FMT_YUV420P)

    # The worker only renders while `render` is on, and training would
    # change the scene from frame to frame.
    w = gui.worker
    gui.ui_state.train[] = false
    gui.ui_state.render[] = true
    w.train[] = false
    w.render[] = true

    c.is_rendering = true
    request_frame!(c, gui)
    return
end

# Point the camera at the current path pose & ask the worker for it.
function request_frame!(c::CaptureMode, gui)
    k = current_pose(c.camera_path)
    set_c2w!(gui.camera, NU.get_rotation(k), k.t)
    c.pending_version = publish_view!(gui)
    return
end

"""
Advance the capture by one frame, if the worker has rendered the pose we are waiting for.
Called every UI frame while a capture is in progress.
"""
function poll_capture!(c::CaptureMode, gui)
    c.is_rendering || return

    frame = fetch_frame(gui.worker, c.pending_version;
        width=c.width, height=c.height)
    frame ≡ nothing && return # Worker has not caught up yet.

    write_frame!(c, frame)
    advance!(c.camera_path, get_time_step(c.camera_path, c.steps_ref[]))
    if c.frame_index ≥ c.total_frames
        stop_capture!(c)
    else
        request_frame!(c, gui)
    end
    return
end

"""
Append a worker frame to the video (and to the frame images, if enabled).
Worker frames are `(3, width, height)` & bottom-up, as OpenGL wants them,
so they are flipped back into image order here.
"""
function write_frame!(c::CaptureMode, frame::Array{Float32, 3})
    image = RGB{N0f8}.(to_image(reverse(frame; dims=3)))
    c.frame_index += 1

    if c.save_frames[]
        image_file = "gsp-$(lpad(c.frame_index, 6, '0')).png"
        save(joinpath(save_directory(c), "images", image_file), image)
    end
    write(c.writer, image)
    return
end

# Contents of the `Capture` tab (see `handle_ui!`).
function capture_ui!(c::CaptureMode, gui)
    CImGui.TextWrapped(
        "Renders a video along a camera path, with the same render " *
        "settings & resolution as the scene view.")
    CImGui.TextWrapped(
        "Press V to add the current camera to the path " *
        "(at least 2 keyframes are required).")
    CImGui.Text("N Keyframes: $(length(c.camera_path))")

    if c.is_rendering
        CImGui.TextWrapped(
            "Capturing. Please wait...\nThe scene view is not resizable " *
            "during this stage.")

        CImGui.PushStyleColor(CImGui.ImGuiCol_PlotHistogram, CImGui.HSV(0.61f0, 1.0f0, 1f0))
        CImGui.ProgressBar(c.frame_index / c.total_frames, CImGui.ImVec2(-1f0, 0f0),
            "$(c.frame_index) / $(c.total_frames)")
        CImGui.PopStyleColor()

        if CImGui.Button("Cancel", CImGui.ImVec2(-1, 0))
            stop_capture!(c)
        end
        return
    end

    if CImGui.CollapsingHeader("Video Settings", CIM_HEADER)
        CImGui.PushItemWidth(-100)
        CImGui.SliderInt("Lerp Steps", c.steps_ref, 1, 60, "%d / 60")
        CImGui.PushItemWidth(-100)
        CImGui.SliderInt("Frame rate", c.framerate_ref, 1, 60, "%d")

        CImGui.Text("Save Directory:")
        CImGui.PushItemWidth(-1)
        CImGui.InputText("##save-dir", pointer(c.save_dir), length(c.save_dir))
        if CImGui.Button("Browse...", CImGui.ImVec2(-1, 0))
            save_dir = pick_folder(homedir()) # Empty when cancelled.
            isempty(save_dir) || set_save_directory!(c, save_dir)
        end

        CImGui.PushItemWidth(-100)
        CImGui.Checkbox("Save frames", c.save_frames)
        CImGui.SetItemTooltip(
            "Also write every frame as a `.png` into an `images` " *
            "subdirectory, next to the video.")
    end

    CImGui.BeginTable("##capture-buttons-table", 2)
    CImGui.TableNextRow()
    CImGui.TableNextColumn()

    # Capturing an empty scene would just write a blank video.
    can_capture = length(c.camera_path) ≥ 2 && gui.worker.n_gaussians[] > 0
    can_capture || disabled_begin()
    if CImGui.Button("Capture", CImGui.ImVec2(-1, 0))
        start_capture!(c, gui)
    end
    can_capture || disabled_end()
    CImGui.SetItemTooltip("Stops training for the duration of the capture.")

    CImGui.TableNextColumn()
    red_button_begin()
    if CImGui.Button("Clear Path", CImGui.ImVec2(-1, 0))
        empty!(c.camera_path)
    end
    red_button_end()
    CImGui.EndTable()
    return
end
