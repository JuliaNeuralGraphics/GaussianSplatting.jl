# Immutable view request published by the UI thread.
# The camera is a deepcopy: the worker treats it as read-only.
struct ViewSnapshot
    camera::Camera
    sh_degree::Int # -1 means use value from the model.
    mode::Int # 0 - color, 1 - depth.
end

# Host-side frame handed from the worker to the UI thread.
mutable struct FrameBuffer
    data::Array{Float32, 3} # (3, width, height), ready for GL upload.
    width::Int
    height::Int
end

FrameBuffer(width::Int, height::Int) =
    FrameBuffer(zeros(Float32, 3, width, height), width, height)

"""
What the worker is currently doing, published to the UI so it can show
a spinner while an operation blocks for long: the first render and the
first training step JIT-compile all GPU kernels (~30 s), during which
the app would otherwise look frozen.
"""
@enum WorkerActivity::Int32 begin
    ActivityIdle
    ActivityTraining
    ActivityRendering
    ActivityLoadingScene
    ActivitySaving
end

function activity_label(activity::WorkerActivity)
    activity ≡ ActivityTraining && return "Training step"
    activity ≡ ActivityRendering && return "Rendering"
    activity ≡ ActivityLoadingScene && return "Installing scene"
    activity ≡ ActivitySaving && return "Saving checkpoint"
    return "Working"
end

"""
Background worker owning all GPU work: training steps and view
rasterization run sequentially on a single task, so densification
(which reallocates the `GaussianModel` arrays) can never race a
concurrent render and all GPU work stays on one task-local stream.

The UI thread owns the OpenGL context, `gui.camera` & the UI state.
It communicates with the worker only through this struct:
- UI → worker: a `ViewSnapshot` (camera + render settings) under `lock`,
  atomic control flags and a `commands` channel for one-shot operations
  (scene install, checkpoint save, orbit-target picking).
- worker → UI: rendered host frames via a double buffer swapped under `lock`,
  atomic stats and error / pick-result slots under `lock`.
"""
mutable struct RenderWorker
    task::Maybe{Task}
    lock::ReentrantLock # Guards: snapshot, front/back swap, error_msg, pick_result.
    wakeup::Base.Event # Autoreset: wakes the worker when it idles.
    commands::Channel{Any}

    # UI → worker (under `lock`).
    snapshot::Maybe{ViewSnapshot}
    snapshot_version::UInt64

    # UI → worker control flags.
    train::Threads.Atomic{Bool}
    densify::Threads.Atomic{Bool}
    render::Threads.Atomic{Bool}

    running::Threads.Atomic{Bool}

    # worker → UI frames (under `lock`).
    front::FrameBuffer
    back::FrameBuffer
    has_new_frame::Bool

    # worker → UI stats.
    loss::Threads.Atomic{Float32}
    step::Threads.Atomic{Int}
    n_gaussians::Threads.Atomic{Int}

    # worker → UI activity (see `with_activity` & `busy_status`).
    activity::Threads.Atomic{Int32} # A `WorkerActivity`.
    activity_since::Threads.Atomic{Float64} # `time()` when it started.

    # worker → UI, rare (under `lock`).
    error_msg::String # Empty when no error.
    pick_result::Maybe{SVector{3, Float32}}

    # Worker-local: the snapshot of the last render, so commands that
    # inspect `rast.image` (orbit picking) know the matching camera.
    last_snapshot::Maybe{ViewSnapshot}
end

function RenderWorker(; width::Int, height::Int)
    RenderWorker(
        nothing, ReentrantLock(), Base.Event(true), Channel{Any}(32),
        nothing, UInt64(0),
        # train, densify, render
        Threads.Atomic{Bool}(false), Threads.Atomic{Bool}(true), Threads.Atomic{Bool}(true),
        # running
        Threads.Atomic{Bool}(false),
        FrameBuffer(width, height), FrameBuffer(width, height), false,
        Threads.Atomic{Float32}(0f0), Threads.Atomic{Int}(0),
        Threads.Atomic{Int}(0),
        # activity, activity_since
        Threads.Atomic{Int32}(Int32(ActivityIdle)), Threads.Atomic{Float64}(0.0),
        "", nothing, nothing)
end

"""
Run `f` while publishing `activity` to the UI.
`activity_since` is stored before the activity becomes visible, so that
the UI can never pair a just-started activity with a stale (older)
timestamp, which would flash the spinner.
"""
function with_activity(f, w::RenderWorker, activity::WorkerActivity)
    w.activity_since[] = time()
    w.activity[] = Int32(activity)
    try
        return f()
    finally
        w.activity[] = Int32(ActivityIdle)
    end
end

"""
What the worker is busy with and for how long (in seconds);
`nothing` when it is idle.
Read from the UI thread, hence only atomics are touched.
"""
function busy_status(w::RenderWorker)
    activity = WorkerActivity(w.activity[])
    activity ≡ ActivityIdle && return nothing
    return (; activity, elapsed=time() - w.activity_since[])
end

"""
The worker needs a thread that is not occupied by the GLFW render loop.
`Threads.@spawn` targets the `:default` pool, so when the main thread
runs in the `:interactive` pool (`-t N,1`) one default thread suffices,
otherwise two are needed.
"""
function check_worker_threads()
    n_default = Threads.nthreads(:default)
    ok = Threads.threadpool() == :interactive ? n_default ≥ 1 : n_default ≥ 2
    ok || error(
        "GaussianSplatting GUI requires at least 2 Julia threads " *
        "to run training in the background. " *
        "Restart Julia with e.g. `julia -t 2,1` or `julia -t auto`.")
    return
end

function start_worker!(gui)
    # Scene arrays may have been created on this task's stream (GSGUI
    # constructors): retire that work before the worker task, which
    # runs on its own task-local stream, touches them.
    KA.synchronize(get_backend(gui.rasterizer))

    w = gui.worker
    w.running[] = true
    w.task = Threads.@spawn worker_loop!(gui, w)
    return
end

function stop_worker!(w::RenderWorker)
    w.running[] = false
    notify(w.wakeup)
    task = w.task
    w.task = nothing
    task ≡ nothing || wait(task)
    close(w.commands)
    return
end

function submit!(w::RenderWorker, cmd::Tuple)
    put!(w.commands, cmd)
    notify(w.wakeup)
    return
end

function set_error!(w::RenderWorker, msg::String)
    lock(w.lock) do
        w.error_msg = msg
    end
    return
end

function take_error!(w::RenderWorker)::Maybe{String}
    lock(w.lock) do
        isempty(w.error_msg) && return nothing
        msg = w.error_msg
        w.error_msg = ""
        return msg
    end
end

function take_pick_result!(w::RenderWorker)::Maybe{SVector{3, Float32}}
    lock(w.lock) do
        result = w.pick_result
        w.pick_result = nothing
        return result
    end
end

# Publish the current camera & render settings for the worker to render.
function publish_view!(gui)
    w = gui.worker
    snap = ViewSnapshot(
        deepcopy(gui.camera),
        Int(gui.ui_state.sh_degree[]),
        Int(gui.ui_state.selected_mode[]))
    lock(w.lock) do
        w.snapshot = snap
        w.snapshot_version += 1
    end
    notify(w.wakeup)
    return
end

# Push the UI toggles to the worker flags (after `reset_ui!`).
function sync_worker_flags!(gui)
    w = gui.worker
    w.train[] = gui.ui_state.train[]
    w.densify[] = gui.ui_state.densify[]
    w.render[] = gui.ui_state.render[]
    notify(w.wakeup)
    return
end

# Upload the latest worker-rendered frame to the GL render surface.
# Runs on the UI thread: the only place where worker output meets GL.
function upload_frame!(gui)
    w = gui.worker
    (; width, height) = resolution(gui.camera)
    lock(w.lock) do
        w.has_new_frame || return
        frame = w.front
        # The surface was resized after this frame was rendered: drop
        # it, the worker renders a matching one from the next snapshot.
        (frame.width == width && frame.height == height) || return

        NGL.set_data!(gui.render_state.surface, frame.data)
        w.has_new_frame = false
        return
    end
    return
end

function worker_loop!(gui, w::RenderWorker)
    last_version = UInt64(0)
    last_view_time = 0.0
    while w.running[]
        try
            # A scene install invalidates the last rendered frame.
            if drain_commands!(gui, w)
                last_version = UInt64(0)
            end

            did_train = false
            trainer = gui.trainer
            if w.train[] && trainer ≢ nothing
                trainer.densify = w.densify[]
                try
                    loss = with_activity(w, ActivityTraining) do
                        step!(trainer)
                    end
                    w.loss[] = loss
                    w.step[] = trainer.step
                    w.n_gaussians[] = length(gui.gaussians)
                    did_train = true
                catch err
                    # E.g. non-finite loss:
                    # stop training instead of killing the worker, the scene stays viewable.
                    w.train[] = false
                    set_error!(w, "Training step failed, stopping training. See logs for details.")
                    @error "Training step failed, stopping training." exception=(err, catch_backtrace())
                end
            end

            snap, version = lock(w.lock) do
                w.snapshot, w.snapshot_version
            end

            did_render = false
            if w.render[] && snap ≢ nothing
                # Re-render on camera/settings change;
                # during training refresh the view at ≤ 10 FPS to not starve `step!`.
                if version != last_version || (did_train && time() - last_view_time > 0.1)
                    with_activity(w, ActivityRendering) do
                        render_view!(gui, w, snap)
                    end
                    last_version = version
                    last_view_time = time()
                    did_render = true
                end
            end

            if !(did_train || did_render)
                wait(w.wakeup)
            end
        catch err
            set_error!(w, "Render worker error. See logs for details.")
            @error "Render worker error:" exception=(err, catch_backtrace())
            wait(w.wakeup) # Avoid hot-spinning on a persistent error.
        end
    end
    return
end

# Returns `true` if a command replaced the scene.
function drain_commands!(gui, w::RenderWorker)
    scene_changed = false
    while isready(w.commands)
        cmd = take!(w.commands)::Tuple
        try
            scene_changed |= with_activity(w, command_activity(cmd[1]::Symbol)) do
                handle_command!(gui, w, cmd)
            end
        catch err
            set_error!(w, "`$(first(cmd))` failed. See logs for details.")
            @error "Worker command failed:" cmd=first(cmd) exception=(err, catch_backtrace())
        end
    end
    return scene_changed
end

# What the UI shows while a command is being handled.
function command_activity(tag::Symbol)
    (tag ≡ :install_scene || tag ≡ :install_bson) && return ActivityLoadingScene
    tag ≡ :save_bson && return ActivitySaving
    return ActivityRendering # `:pick_orbit` reads the rendered depth.
end

"""
Handle commands send by GUI.
Return `true` if any of the commands requires re-rendering of the scene.
"""
function handle_command!(gui, w::RenderWorker, cmd::Tuple)
    tag = cmd[1]::Symbol
    if tag ≡ :install_scene
        loaded = cmd[2]
        gui.gaussians = loaded.gaussians
        gui.rasterizer = loaded.gui_rasterizer
        gui.trainer = loaded.trainer
        w.loss[] = 0f0
        w.step[] = loaded.trainer.step
        w.n_gaussians[] = length(loaded.gaussians)
        return true
    elseif tag ≡ :install_bson
        gaussians = cmd[2]
        gui.gaussians = gaussians
        gui.trainer = nothing # Viewer-only mode.
        w.loss[] = 0f0
        w.step[] = 0
        w.n_gaussians[] = length(gaussians)
        return true
    elseif tag ≡ :save_bson
        # Reads GPU arrays: must be ordered with training, hence here.
        gui.trainer ≡ nothing || save_state(gui.trainer, cmd[2]::String)
        return false
    elseif tag ≡ :pick_orbit
        handle_pick!(gui, w, cmd[2]::Int, cmd[3]::Int)
        return true
    end
    error("Unknown worker command: `$tag`.")
end

# Render the view described by `snap` into the back buffer & swap.
function render_view!(gui, w::RenderWorker, snap::ViewSnapshot)
    camera = snap.camera
    (; width, height) = resolution(camera)

    rast = gui.rasterizer
    if size(rast.image)[2:3] != (width, height)
        kab = get_backend(rast)
        # TODO free the old one before creating new one.
        gui.rasterizer = rast = GaussianRasterizer(kab, camera; fused=true, mode=:rgbd)
    end

    back = w.back
    if back.width != width || back.height != height
        back.data = Array{Float32, 3}(undef, 3, width, height)
        back.width, back.height = width, height
    end

    gs = gui.gaussians
    if gs ≡ nothing || length(gs) == 0
        # Empty scene (no dataset loaded yet): display background color.
        fill!(back.data, 0f0)
    else
        sh_degree = snap.sh_degree == -1 ? gs.sh_degree : snap.sh_degree
        rast(
            gs.points, gs.opacities, gs.scales,
            gs.rotations, gs.features_dc, gs.features_rest;
            camera, sh_degree)
        tex = snap.mode == 1 ? gl_depth(rast) : gl_texture(rast)
        # `tex` is the rasterizer-owned host buffer, overwritten by the next render,
        # so copy it out before publishing.
        copyto!(back.data, tex)
    end

    w.last_snapshot = snap
    lock(w.lock) do
        w.front, w.back = w.back, w.front
        w.has_new_frame = true
    end
    return
end

"""
Pick a new orbiting target by unprojecting the rendered depth under
the `(px, py)` pixel (1-based, top-left origin).
Depth is averaged over a small window around the pixel to be robust
to outliers at fuzzy silhouettes; background pixels (nothing rendered
there, so the blended depth is ≈ 0) are excluded.
Uses the camera of the last-rendered snapshot, matching the contents
of `rast.image`; the result is published to `pick_result`.
"""
function handle_pick!(gui, w::RenderWorker, px::Integer, py::Integer)
    snap = w.last_snapshot
    snap ≡ nothing && return

    gs = gui.gaussians
    (gs ≡ nothing || length(gs) == 0) && return

    rast = gui.rasterizer
    rast.mode == :rgbd || return

    camera = snap.camera
    (; width, height) = resolution(camera)
    (1 ≤ px ≤ width && 1 ≤ py ≤ height) || return

    δ = 4 # Window half-size.
    depths = Array(@view(rast.image[4,
        max(1, px - δ):min(width, px + δ),
        max(1, py - δ):min(height, py + δ)]))
    valid = depths .> 1f-2 # Near plane: rejects background pixels.
    any(valid) || return
    z = mean(depths[valid])

    # Unproject through the camera intrinsics
    # (COLMAP frame: x right, y down, z forward).
    fx, fy = camera.intrinsics.focal
    cx = camera.intrinsics.principal[1] * width
    cy = camera.intrinsics.principal[2] * height
    p_cam = SVector{3, Float32}(
        (px - 0.5f0 - cx) * z / fx,
        (py - 0.5f0 - cy) * z / fy,
        z)
    R = SMatrix{3, 3, Float32}(@view(camera.c2w[1:3, 1:3]))
    target = R * p_cam .+ view_pos(camera)

    lock(w.lock) do
        w.pick_result = target
    end
    return
end
