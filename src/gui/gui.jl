# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
include("ui_state.jl")
include("render_state.jl")
include("worker.jl")
include("camera_path.jl")
include("capture_mode.jl")

const CIM_HEADER =
    CImGui.ImGuiTreeNodeFlags_CollapsingHeader |
    CImGui.ImGuiTreeNodeFlags_DefaultOpen

function red_button_begin()
    CImGui.PushStyleColor(CImGui.ImGuiCol_Button, CImGui.HSV(0f0, 0.6f0, 0.6f0))
    CImGui.PushStyleColor(CImGui.ImGuiCol_ButtonHovered, CImGui.HSV(0f0, 0.7f0, 0.7f0))
    CImGui.PushStyleColor(CImGui.ImGuiCol_ButtonActive, CImGui.HSV(0f0, 0.7f0, 0.7f0))
end

function red_button_end()
    CImGui.PopStyleColor(3)
end

function disabled_begin()
    CImGui.igPushItemFlag(CImGui.ImGuiItemFlags_Disabled, true)
    alpha = unsafe_load(CImGui.GetStyle().Alpha) * 0.5f0
    CImGui.PushStyleVar(CImGui.ImGuiStyleVar_Alpha, alpha)
end

function disabled_end()
    CImGui.PopStyleVar()
    CImGui.igPopItemFlag()
end

# How long the worker must be busy with the same operation before the
# UI shows a spinner: long enough not to flicker on regular renders &
# training steps, short enough to appear before the app feels stuck.
const SPINNER_DELAY = 0.5
# Past this point the wait is dominated by GPU kernel compilation
# (or a densification pass): explain it instead of just spinning.
const SPINNER_HINT_DELAY = 3.0

# Visible rows of the `Camera view` list; the rest is scrolled to.
const VIEW_LIST_ROWS = 8

"""
Rotating arc, animated off `CImGui.GetTime()` (wall clock), so it keeps
spinning at a steady rate regardless of the UI frame rate.
"""
function draw_spinner(draw_list, center, radius::Float32, thickness::Float32, color)
    t = Float32(CImGui.GetTime())
    from = 3f0 * t
    # Pulsing arc length: reads as motion even when the arc is symmetric.
    to = from + 1.1f0 * Float32(π) + 0.6f0 * Float32(π) * sin(2f0 * t)

    CImGui.PathClear(draw_list)
    CImGui.PathArcTo(draw_list, center, radius, from, to, 32)
    CImGui.PathStroke(draw_list, color, 0, thickness)
    return
end

# Inline spinner widget: occupies layout space like any other item.
function spinner!(; radius::Float32 = 7f0, thickness::Float32 = 3f0)
    pos = CImGui.GetCursorScreenPos()
    size = 2f0 * (radius + thickness)
    draw_spinner(
        CImGui.GetWindowDrawList(),
        (pos.x + 0.5f0 * size, pos.y + 0.5f0 * size),
        radius, thickness, CImGui.GetColorU32(CImGui.ImGuiCol_Text))
    CImGui.Dummy(CImGui.ImVec2(size, size))
    return
end

"""
Compact `<spinner> Rendering...` status line for the controls window.
Always occupies exactly one line, so the widgets below do not jump when
the worker goes busy / idle.
"""
function worker_busy_line!(w::RenderWorker)
    radius, thickness = 6f0, 2.5f0
    status = busy_status(w)
    if status ≢ nothing && status.elapsed ≥ SPINNER_DELAY
        spinner!(; radius, thickness)
        CImGui.SameLine()
        CImGui.Text("$(activity_label(status.activity))...")
    else
        CImGui.Dummy(CImGui.ImVec2(0f0, 2f0 * (radius + thickness)))
    end
    return
end

function is_mouse_in_ui()
    CImGui.IsMousePosValid() && unsafe_load(CImGui.GetIO().WantCaptureMouse)
end

function enable_docking!()
    io = CImGui.GetIO()
    io.ConfigFlags = unsafe_load(io.ConfigFlags) | CImGui.ImGuiConfigFlags_DockingEnable
    return
end

function dockspace!()
    # Passthru central node keeps the scene visible & interactive where no window is docked.
    return CImGui.DockSpaceOverViewport(
        0, CImGui.GetMainViewport(),
        CImGui.ImGuiDockNodeFlags_PassthruCentralNode)
end

function look_at(position, target, up)
    Z = normalize(position - target)
    X = normalize(normalize(up) × Z)
    Y = Z × X

    SMatrix{4, 4, Float32}(
        X[1], Y[1], Z[1], 0f0,
        X[2], Y[2], Z[2], 0f0,
        X[3], Y[3], Z[3], 0f0,
        X ⋅ -position, Y ⋅ -position, Z ⋅ -position, 1f0)
end

# Extend GL a bit.
function NGL.look_at(c::Camera)
    look_at(view_pos(c), look_at(c), -view_up(c))
end

function NGL.perspective(
    c::Camera; near::Float32 = 0.1f0, far::Float32 = 100f0,
)
    fov_xy = NU.focal2fov.(c.intrinsics.resolution, c.intrinsics.focal)
    NGL.perspective(fov_xy..., near, far)
end

function NU.CameraKeyframe(c::Camera)
    R, t = c.c2w[1:3, 1:3], c.c2w[1:3, 4]
    q = QuatRotation{Float32}(R)
    NU.CameraKeyframe(QuaternionF32(q.w, q.x, q.y, q.z), t)
end

@enum Screen begin
    MainScreen
    CaptureScreen
end

# Fields are non-concrete so that a dataset can be loaded at runtime
# (`trainer` goes from `nothing` to a `Trainer`, gaussians & rasterizer
# are replaced): see `load_dataset!`.
mutable struct GSGUI
    context::NGL.Context
    screen::Screen
    frustum::NGL.Frustum
    render_state::RenderState
    ui_state::UIState
    control_settings::ControlSettings

    capture_mode::CaptureMode

    camera::Camera
    # Owned by the render worker while it runs: the UI thread must not
    # touch the GPU state behind these (see `RenderWorker`).
    # Reads of the references themselves (e.g. `viewer_only`) are benign: they
    # only change via installs the UI itself initiated.
    gaussians::Maybe{GaussianModel}
    rasterizer::GaussianRasterizer
    trainer::Maybe{Trainer}

    worker::RenderWorker
end

const GSGUI_REF::Ref{GSGUI} = Ref{GSGUI}()

function resize_callback(_, width, height)
    (width == 0 || height == 0) && return # Window minimized.

    NGL.set_viewport(width, height)
    isassigned(GSGUI_REF) || return

    # Render resolution follows the `Scene` window size, not the OS window:
    # the dock layout adjusts and `scene_window!` picks up the new size.
    GSGUI_REF[].render_state.need_render = true
    return
end

# Resize render resolution to match the `Scene` window content size.
function resize_scene!(gui::GSGUI; width::Int, height::Int)
    NGL.resize!(gui.render_state.surface; width, height)
    for attachment in values(gui.render_state.framebuffer.attachments)
        NGL.resize!(attachment; width, height)
    end

    set_resolution!(gui.camera; width, height)
    # The worker rebuilds its rasterizer from the camera resolution of
    # the next published snapshot.
    gui.render_state.need_render = true
    return
end

# Viewer-only mode.
function GSGUI(kab, gaussians::Maybe{GaussianModel}, camera::Camera; gl_kwargs...)
    check_worker_threads()
    NGL.init(3, 2)
    context = NGL.Context("GaussianSplatting.jl"; gl_kwargs...)
    NGL.set_resize_callback!(context, resize_callback)

    font_file = joinpath(pkgdir(CImGui), "fonts", "Roboto-Medium.ttf")
    fonts = unsafe_load(CImGui.GetIO().Fonts)
    CImGui.AddFontFromFileTTF(fonts, font_file, 16)

    enable_docking!()

    # Set up renderer.
    set_resolution!(camera; (;
        width=16 * cld(context.width, 16),
        height=16 * cld(context.height, 16))...)
    rasterizer = GaussianRasterizer(kab, camera)

    render_state = RenderState(;
        surface=NGL.RenderSurface(; internal_format=GL_RGB32F, data_type=GL_FLOAT, resolution(camera)...),
        framebuffer=NGL.Framebuffer(; resolution(camera)...))
    control_settings = ControlSettings()
    ui_state = UIState()
    ui_state.max_sh_degree = gaussians ≡ nothing ? 0 : gaussians.max_sh_degree

    capture_mode = CaptureMode()

    worker = RenderWorker(; resolution(camera)...)
    worker.n_gaussians[] = gaussians ≡ nothing ? 0 : length(gaussians)

    trainer = nothing
    gsgui = GSGUI(
        context, MainScreen, NGL.Frustum(), render_state, ui_state,
        control_settings, capture_mode, camera,
        gaussians, rasterizer, trainer, worker)
    GSGUI_REF[] = gsgui
    return gsgui
end

# Densification strategies selectable in the UI.
const STRATEGIES = (:default, :mcmc)

# Training mode.
function GSGUI(kab, dataset_path::String, scale::Int;
    strategy::Symbol = :default, use_depth_loss::Bool = true,
    use_bilateral_grid::Bool = false, use_normal_loss::Bool = false,
    gl_kwargs...,
)
    check_worker_threads()
    NGL.init(3, 2)
    context = NGL.Context("GaussianSplatting.jl"; gl_kwargs...)
    NGL.set_resize_callback!(context, resize_callback)

    font_file = joinpath(pkgdir(CImGui), "fonts", "Roboto-Medium.ttf")
    fonts = unsafe_load(CImGui.GetIO().Fonts)
    CImGui.AddFontFromFileTTF(fonts, font_file, 16)

    enable_docking!()

    dataset = ColmapDataset(kab, dataset_path; scale, holdout=0)
    camera = dataset.train_cameras[1]

    opt_params = OptimizationParams(;
        use_depth_loss, use_bilateral_grid, use_normal_loss)
    gaussians = GaussianModel(dataset.points, dataset.colors, dataset.scales;
        isotropic=false, max_sh_degree=3)
    rasterizer = GaussianRasterizer(kab, camera;
        mode=training_rasterizer_mode(opt_params))
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params;
        strategy=create_strategy(strategy, gaussians))

    # Set-up separate renderer camera & rasterizer.
    camera = deepcopy(camera)
    set_resolution!(camera; (;
        width=16 * cld(context.width, 16),
        height=16 * cld(context.height, 16))...)
    gui_rasterizer = GaussianRasterizer(kab, camera; mode=:rgbd)

    render_state = RenderState(;
        surface=NGL.RenderSurface(;
            internal_format=GL_RGB32F, data_type=GL_FLOAT,
            resolution(camera)...),
        framebuffer=NGL.Framebuffer(; resolution(camera)...))
    control_settings = ControlSettings()
    control_settings.up_vec = estimate_up_vec(dataset.train_cameras)
    ui_state = UIState()
    ui_state.max_sh_degree = gaussians.max_sh_degree
    ui_state.is_mcmc = trainer.strategy isa MCMCStrategy

    capture_mode = CaptureMode()

    worker = RenderWorker(; resolution(camera)...)
    worker.n_gaussians[] = length(gaussians)

    gsgui = GSGUI(
        context, MainScreen, NGL.Frustum(), render_state, ui_state,
        control_settings, capture_mode, camera,
        gaussians, gui_rasterizer, trainer, worker)
    GSGUI_REF[] = gsgui
    return gsgui
end

viewer_only(gui::GSGUI) = isnothing(gui.trainer)

"""
Load a COLMAP dataset: new gaussians, trainer and rasterizers.
`width` & `height` specify the render resolution for the GUI camera.

Runs on a background thread (see `open_dataset_modal!`), so it must
not touch OpenGL state: the results are applied on the render thread
in `apply_dataset!`.
"""
function load_dataset(kab, dataset_path::String;
    scale::Int, width::Int, height::Int, strategy::Symbol = :default,
    use_depth_loss::Bool = true, use_bilateral_grid::Bool = false,
    use_normal_loss::Bool = false,
)
    dataset = ColmapDataset(kab, dataset_path; scale, holdout=0)
    camera = dataset.train_cameras[1]

    opt_params = OptimizationParams(;
        use_depth_loss, use_bilateral_grid, use_normal_loss)
    gaussians = GaussianModel(dataset.points, dataset.colors, dataset.scales;
        isotropic=false, max_sh_degree=3)
    rasterizer = GaussianRasterizer(kab, camera;
        mode=training_rasterizer_mode(opt_params))
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params;
        strategy=create_strategy(strategy, gaussians))

    # Set-up separate renderer camera & rasterizer.
    camera = deepcopy(camera)
    set_resolution!(camera; width, height)
    # TODO free the old one before creating new one.
    gui_rasterizer = GaussianRasterizer(kab, camera; mode=:rgbd)

    up_vec = estimate_up_vec(dataset.train_cameras)
    # H2D copies above run on this task's stream: make sure they are
    # done before the worker task touches the new arrays.
    KA.synchronize(kab)
    return (; camera, gaussians, gui_rasterizer, trainer, up_vec)
end

# Replace the current scene, keeping the GL context & render surface.
# UI-side part runs here; the GPU state is installed by the worker.
function apply_dataset!(gui::GSGUI, loaded)
    gui.camera = loaded.camera
    # Yaw rotates around the estimated scene up: keeps the horizon level.
    gui.control_settings.up_vec = loaded.up_vec

    reset_ui!(gui.ui_state)
    gui.ui_state.max_sh_degree = loaded.gaussians.max_sh_degree
    gui.ui_state.is_mcmc = loaded.trainer.strategy isa MCMCStrategy
    sync_worker_flags!(gui)

    submit!(gui.worker, (:install_scene, loaded))
    gui.render_state.need_render = true
    return
end

"""
Load a `.bson` model checkpoint for viewer-only mode (no trainer).

Runs on a background thread (see `menu_bar!`), so it must not touch
OpenGL state: the results are applied on the render thread in
`apply_bson!`.
"""
function load_bson(kab, state_file::String)
    θ = load_checkpoint(state_file)
    gaussians = GaussianModel(kab)
    set_from_bson!(gaussians, θ[:gaussians])
    # H2D copies above run on this task's stream: make sure they are
    # done before the worker task touches the new arrays.
    KA.synchronize(kab)
    return (; camera=θ[:camera]::Camera, gaussians)
end

# Replace the current scene with a loaded checkpoint (viewer-only).
function apply_bson!(gui::GSGUI, loaded)
    # Keep the current render resolution.
    camera = loaded.camera
    set_resolution!(camera; resolution(gui.camera)...)
    gui.camera = camera

    reset_ui!(gui.ui_state)
    gui.ui_state.max_sh_degree = loaded.gaussians.max_sh_degree
    gui.ui_state.is_mcmc = false
    sync_worker_flags!(gui)

    submit!(gui.worker, (:install_bson, loaded.gaussians))
    gui.render_state.need_render = true
    return
end

function poll_bson_load!(gui::GSGUI)
    task = gui.ui_state.bson_load_task
    (task ≡ nothing || !istaskdone(task)) && return
    gui.ui_state.bson_load_task = nothing
    try
        apply_bson!(gui, fetch(task))
    catch err
        @error "Failed to load BSON checkpoint:" exception=(err, catch_backtrace())
    end
    return
end

"""
Whether a scene is loaded, i.e. whether `Close Scene` has anything to do.
Reads the worker's atomic gaussian count rather than `gui.gaussians`, whose
arrays the worker replaces on every densification.
"""
scene_loaded(gui::GSGUI) = gui.trainer ≢ nothing || gui.worker.n_gaussians[] > 0

"""
Unload the current scene, freeing the memory it holds.
UI-side part runs here; the GPU state is released by the worker
(`handle_close_scene!`), leaving an empty scene behind.
"""
function close_scene!(gui::GSGUI)
    reset_ui!(gui.ui_state)
    gui.ui_state.max_sh_degree = 0
    gui.ui_state.is_mcmc = false
    sync_worker_flags!(gui)

    submit!(gui.worker, (:close_scene,))
    gui.render_state.need_render = true
    return
end

function reset_ui!(ui_state::UIState)
    ui_state.train[] = false
    ui_state.densify[] = true
    ui_state.loss = 0f0
    ui_state.selected_view[] = 0
    ui_state.selected_mode[] = 0
    ui_state.sh_degree[] = -1
    ui_state.draw_cameras[] = false
    ui_state.worker_error = ""
    return
end

function menu_bar!(gui::GSGUI)
    CImGui.BeginMainMenuBar() || return

    if CImGui.BeginMenu("File")
        if CImGui.MenuItem("Open Dataset...")
            gui.ui_state.open_dataset_popup = true
        end

        CImGui.Separator()

        # Viewer-only mode: no trainer.
        if CImGui.MenuItem("Open BSON...")
            state_file = pick_file(homedir(); filterlist="bson") # Empty when cancelled.
            if !isempty(state_file)
                # Only the backend type is read here: safe from the UI thread.
                kab = get_backend(gui.rasterizer)
                gui.ui_state.bson_load_task =
                    Threads.@spawn load_bson(kab, state_file)
            end
        end

        CImGui.Separator()

        # Saving needs a trainer: it stores optimizers & training step.
        if CImGui.MenuItem("Save BSON...", C_NULL, false, !viewer_only(gui))
            state_file = save_file(homedir(); filterlist="bson") # Empty when cancelled.
            if !isempty(state_file)
                endswith(state_file, ".bson") || (state_file *= ".bson")
                # Saving reads GPU arrays: run on the worker so it is
                # ordered with training steps.
                submit!(gui.worker, (:save_bson, state_file))
            end
        end

        CImGui.Separator()

        if CImGui.MenuItem("Close Scene", C_NULL, false, scene_loaded(gui))
            close_scene!(gui)
        end
        CImGui.SetItemTooltip(
            "Unload the gaussians & the dataset, freeing their GPU memory.")
        CImGui.EndMenu()
    end

    CImGui.EndMainMenuBar()
    return
end

"""
Modal window for selecting a COLMAP dataset folder & its scale.
Opened via the `File` menu; must be submitted at the same ID stack
level as `OpenPopup`, hence outside of `menu_bar!`.
"""
function open_dataset_modal!(gui::GSGUI)
    ui_state = gui.ui_state
    if ui_state.open_dataset_popup
        ui_state.open_dataset_popup = false
        ui_state.dataset_error = ""
        CImGui.OpenPopup("Open Dataset")
    end

    # Center on the viewport.
    viewport = CImGui.GetMainViewport()
    vp_pos, vp_size = unsafe_load(viewport.Pos), unsafe_load(viewport.Size)
    CImGui.SetNextWindowPos(
        (vp_pos.x + 0.5f0 * vp_size.x, vp_pos.y + 0.5f0 * vp_size.y),
        CImGui.ImGuiCond_Appearing, (0.5f0, 0.5f0))

    # Fixed width; height auto-fits the form on appearance & then stays
    # constant, so the window does not shrink when only the loading
    # spinner is displayed.
    CImGui.SetNextWindowSize(
        CImGui.ImVec2(600f0, 0f0), CImGui.ImGuiCond_Appearing)
    CImGui.BeginPopupModal("Open Dataset", C_NULL,
        CImGui.ImGuiWindowFlags_NoResize) || return

    # Loading in progress: show a spinner until the task completes.
    task = ui_state.dataset_load_task
    if task ≢ nothing
        yield() # Let the loading task run when Julia is single-threaded.
        if istaskdone(task)
            ui_state.dataset_load_task = nothing
            try
                apply_dataset!(gui, fetch(task))
                CImGui.CloseCurrentPopup()
            catch err
                ui_state.dataset_error = "Failed to load dataset. See logs for details."
                @error "Failed to load COLMAP dataset:" exception=(err, catch_backtrace())
            end
        else
            spinner!()
            CImGui.SameLine()
            CImGui.Text("Loading dataset. Please wait...")
        end
        CImGui.EndPopup()
        return
    end

    CImGui.Text("Path to COLMAP dataset folder:")
    CImGui.PushItemWidth(400)
    CImGui.InputText("##dataset-path", pointer(ui_state.dataset_path),
        length(ui_state.dataset_path))
    CImGui.PopItemWidth()
    CImGui.SameLine()
    if CImGui.Button("Browse...")
        dataset_path = pick_folder(homedir()) # Empty when cancelled.
        if !isempty(dataset_path)
            ui_state.dataset_path = Vector{UInt8}(dataset_path * "\0"^512)
        end
    end

    CImGui.Text("Scale:")
    for scale in (1, 2, 4, 8)
        CImGui.SameLine()
        if CImGui.RadioButton("$(scale)x", Int(ui_state.dataset_scale[]) == scale)
            ui_state.dataset_scale[] = scale
        end
    end

    CImGui.Text("Densification strategy:")
    for (i, strategy) in enumerate(STRATEGIES)
        CImGui.SameLine()
        if CImGui.RadioButton("$strategy", Int(ui_state.dataset_strategy[]) == i - 1)
            ui_state.dataset_strategy[] = i - 1
        end
    end

    CImGui.Checkbox("Monocular depth supervision", ui_state.dataset_depth_loss)
    if CImGui.IsItemHovered()
        CImGui.SetTooltip(
            "Scale- & shift-invariant loss against depth maps stored next " *
            "to the dataset images.\nSilently disabled when the dataset has " *
            "none.")
    end

    CImGui.Checkbox("Bilateral grid appearance modeling", ui_state.dataset_bilateral_grid)
    if CImGui.IsItemHovered()
        CImGui.SetTooltip(
            "Per-train-image color correction absorbing exposure / " *
            "white-balance drift.\nRecommended for casual (phone) captures.")
    end

    CImGui.Checkbox("Geometry regularization", ui_state.dataset_normal_loss)
    if CImGui.IsItemHovered()
        CImGui.SetTooltip(
            "Depth-normal consistency + flattening along the thinnest axis.\n" *
            "Improves surface geometry at the cost of ~3 extra blended " *
            "channels per step.")
    end

    # Always occupy the error line to keep the window height constant.
    if isempty(ui_state.dataset_error)
        CImGui.Text(" ")
    else
        CImGui.TextColored((1f0, 0.3f0, 0.3f0, 1f0), ui_state.dataset_error)
    end

    CImGui.Separator()
    dataset_path = unsafe_string(pointer(ui_state.dataset_path))
    can_open = isdir(dataset_path)

    can_open || disabled_begin()
    if CImGui.Button("Open", CImGui.ImVec2(120, 0))
        ui_state.dataset_error = ""
        kab = get_backend(gui.rasterizer)
        scale = Int(ui_state.dataset_scale[])
        strategy = STRATEGIES[ui_state.dataset_strategy[] + 1]
        use_depth_loss = ui_state.dataset_depth_loss[]
        use_bilateral_grid = ui_state.dataset_bilateral_grid[]
        use_normal_loss = ui_state.dataset_normal_loss[]
        (; width, height) = resolution(gui.camera)
        ui_state.dataset_load_task = Threads.@spawn load_dataset(
            kab, dataset_path; scale, width, height, strategy,
            use_depth_loss, use_bilateral_grid, use_normal_loss)
    end
    can_open || disabled_end()

    CImGui.SameLine()
    if CImGui.Button("Cancel", CImGui.ImVec2(120, 0))
        CImGui.CloseCurrentPopup()
    end
    CImGui.EndPopup()
    return
end

function launch!(gui::GSGUI)
    start_worker!(gui)
    try
        NGL.render_loop(gui.context) do
            if gui.screen == MainScreen
                loop!(gui)
            else gui.screen == CaptureScreen
                loop!(gui.capture_mode; gui)
            end
            return true
        end
    finally
        stop_worker!(gui.worker)
    end
end

function loop!(gui::GSGUI)
    w = gui.worker
    frame_time = update_time!(gui.render_state)
    NGL.imgui_begin()
    menu_bar!(gui)
    open_dataset_modal!(gui)
    poll_bson_load!(gui)
    dockspace_id = dockspace!()

    # Worker results: stats, errors, orbit-target picks.
    gui.ui_state.loss = w.loss[]
    err = take_error!(w)
    err ≡ nothing || (gui.ui_state.worker_error = err)
    target = take_pick_result!(w)
    target ≡ nothing || (gui.control_settings.orbiting_target = target)

    # Handle controls.
    # `scene_hovered` lags one frame, same as ImGui's `WantCaptureMouse`.
    mouse_in_ui = is_mouse_in_ui() && !gui.ui_state.scene_hovered

    handle_ui!(gui; frame_time)
    if !mouse_in_ui
        controller_id = gui.ui_state.controller_mode[]

        gui.render_state.need_render |= handle_keyboard!(
            gui.control_settings, gui.camera; frame_time, controller_id)
        gui.render_state.need_render |= handle_mouse!(
            gui.control_settings, gui.camera; controller_id)
    end

    # Publish the latest camera & render settings: the worker
    # rasterizes in the background and hands back a host frame,
    # which `scene_window!` → `upload_frame!` displays.
    if gui.render_state.need_render
        gui.render_state.need_render = false
        publish_view!(gui)
    end

    NGL.clear()
    NGL.set_clear_color(0.2, 0.2, 0.2, 1.0)

    # Draw gaussians & other OpenGL objects into the `Scene` window.
    scene_window!(gui, dockspace_id) do
        if !viewer_only(gui) && gui.ui_state.draw_cameras[]
            P = NGL.perspective(gui.camera)
            L = NGL.look_at(gui.camera)

            dataset = gui.trainer.dataset
            for view_id in 1:length(dataset)
                camera = dataset.train_cameras[view_id]
                camera_perspective =
                    NGL.perspective(camera; near=0.1f0, far=0.2f0) *
                    NGL.look_at(camera)
                NGL.draw(gui.frustum, camera_perspective, P, L)
            end
        end
    end

    NGL.imgui_end()
    GLFW.SwapBuffers(gui.context.window)
    GLFW.PollEvents()

    # Load-bearing: only the main thread services Julia's libuv event loop.
    # GPU synchronization on the worker (every device -> host copy) waits on
    # a libuv `AsyncCondition` signalled from a HIP/CUDA callback, so without
    # this the worker blocks forever in its first rasterization, regardless
    # of how many threads are available.
    yield()
    return
end

"""
Scene view as a dockable window: it starts docked into the dockspace's
central node and re-renders at the new resolution when other windows
docking around it change its size.

`extra_draws` is called after the splats are drawn, with the scene
framebuffer still bound, to overlay other OpenGL objects (frustums, etc.).
"""
function scene_window!(
    extra_draws::Function, gui::GSGUI, dockspace_id;
    force_render::Bool = false, allow_resize::Bool = true,
)
    CImGui.SetNextWindowDockID(dockspace_id, CImGui.ImGuiCond_FirstUseEver)
    CImGui.PushStyleVar(
        CImGui.ImGuiStyleVar_WindowPadding, CImGui.ImVec2(0f0, 0f0))
    visible = CImGui.Begin("Scene")
    CImGui.PopStyleVar()

    hovered = false
    if visible && allow_resize
        avail = CImGui.GetContentRegionAvail()
        width = 16 * max(1, floor(Int, avail.x / 16))
        height = 16 * max(1, floor(Int, avail.y / 16))
        res = resolution(gui.camera)
        if width != res.width || height != res.height
            resize_scene!(gui; width, height)
        end
    end

    # if force_render
    #     # Capture mode: synchronous render on the GUI thread
    #     # (the worker is expected to be paused).
    #     render!(gui)
    # else
    upload_frame!(gui)
    # end

    if visible
        draw_scene!(extra_draws, gui)

        res = resolution(gui.camera)
        color = gui.render_state.framebuffer[GL_COLOR_ATTACHMENT0]
        # Flip v: OpenGL textures are bottom-up.
        CImGui.Image(
            CImGui.ImTextureRef(C_NULL, CImGui.ImTextureID(color.id)),
            (Float32(res.width), Float32(res.height)),
            (0f0, 1f0), (1f0, 0f0))
        hovered = CImGui.IsWindowHovered()
        # Drawn after the image, so it is on top of it. Submits no
        # items, leaving the image as the current one for the pick below.
        worker_busy_overlay!(gui, CImGui.GetItemRectMin())

        # Double-click in orbiting mode: set the orbiting target to the
        # point under the cursor. The image is drawn 1:1, so the offset
        # from its top-left corner is the pixel position. The pick runs
        # on the worker (it reads the rendered depth); the result is
        # applied in `loop!` one frame later.
        if gui.ui_state.controller_mode[] == 1 &&
            CImGui.IsItemHovered() && CImGui.IsMouseDoubleClicked(0)

            rect_min = CImGui.GetItemRectMin()
            mouse_pos = CImGui.GetMousePos()
            submit!(gui.worker, (:pick_orbit,
                floor(Int, mouse_pos.x - rect_min.x) + 1,
                floor(Int, mouse_pos.y - rect_min.y) + 1))
        end
    end
    CImGui.End()

    gui.ui_state.scene_hovered = hovered
    return
end

"""
Spinner badge over the `Scene` view while the worker has been busy with
the same operation for longer than `SPINNER_DELAY`.

The UI thread stays responsive while the worker blocks (the scene keeps
showing its last frame), so without this the app looks frozen for the
~30 s the first render / training step spends compiling GPU kernels.

`rect_min` is the top-left screen position of the scene image; the
badge is drawn into the current window's draw list, in its corner.
"""
function worker_busy_overlay!(gui::GSGUI, rect_min)
    status = busy_status(gui.worker)
    status ≡ nothing && return
    (; activity, elapsed) = status
    elapsed < SPINNER_DELAY && return

    label = "$(activity_label(activity))... ($(round(elapsed; digits=1)) s)"
    hint = elapsed < SPINNER_HINT_DELAY ? "" :
        "GPU kernels are compiled on first use: this can take ~30 s."

    pad, radius, thickness = 8f0, 7f0, 3f0
    line_height = CImGui.GetTextLineHeight()
    text_width = CImGui.CalcTextSize(label).x
    text_height = line_height
    if !isempty(hint)
        text_width = max(text_width, CImGui.CalcTextSize(hint).x)
        text_height += line_height
    end

    diameter = 2f0 * radius
    width = 3f0 * pad + diameter + text_width
    height = 2f0 * pad + max(diameter, text_height)

    draw_list = CImGui.GetWindowDrawList()
    x, y = rect_min.x + pad, rect_min.y + pad
    CImGui.AddRectFilled(draw_list, (x, y), (x + width, y + height),
        CImGui.GetColorU32((0f0, 0f0, 0f0, 0.6f0)), 6f0)
    draw_spinner(draw_list,
        (x + pad + radius, y + 0.5f0 * height),
        radius, thickness, CImGui.GetColorU32(CImGui.ImGuiCol_Text))

    text_x = x + 2f0 * pad + diameter
    text_y = y + 0.5f0 * (height - text_height)
    CImGui.AddText(draw_list, (text_x, text_y),
        CImGui.GetColorU32(CImGui.ImGuiCol_Text), label)
    isempty(hint) || CImGui.AddText(draw_list, (text_x, text_y + line_height),
        CImGui.GetColorU32(CImGui.ImGuiCol_TextDisabled), hint)
    return
end

function draw_scene!(extra_draws::Function, gui::GSGUI)
    fb = gui.render_state.framebuffer
    (; width, height) = resolution(gui.camera)

    NGL.bind(fb)
    NGL.set_viewport(width, height)
    NGL.clear()
    NGL.set_clear_color(0.2, 0.2, 0.2, 1.0)

    NGL.draw(gui.render_state.surface)
    NGL.clear(NGL.GL_DEPTH_BUFFER_BIT)
    extra_draws()

    NGL.unbind(fb)
    return
end

function handle_ui!(gui::GSGUI; frame_time)
    w = gui.worker
    if CImGui.Begin("GaussianSplatting")
        (; width, height) = resolution(gui.camera)
        CImGui.Text("Render Resolution: $width x $height")
        CImGui.Text("Backend: $(backend_name(get_backend(gui.rasterizer)))")
        CImGui.Text("GPU Memory: $(Base.format_bytes(w.memory[]))")
        CImGui.SetItemTooltip(
            "Device memory held by the gaussians, the optimizers & the " *
            "rasterizers.\nThe backend's memory pool keeps freed blocks " *
            "around, so the process always holds at least this much.")
        CImGui.Text("Number of Gaussians: $(w.n_gaussians[])")
        worker_busy_line!(w)

        isempty(gui.ui_state.worker_error) || CImGui.TextColored(
            (1f0, 0.3f0, 0.3f0, 1f0), gui.ui_state.worker_error)

        if CImGui.Checkbox("Render", gui.ui_state.render)
            w.render[] = gui.ui_state.render[]
            notify(w.wakeup)
        end

        CImGui.PushItemWidth(-100)
        if CImGui.Combo("Controller", gui.ui_state.controller_mode,
            gui.ui_state.controller_modes,
        ) && gui.ui_state.controller_mode[] == 1
            # Entering orbit mode: place the target in front of the
            # camera, at a scene-sized distance.
            d = viewer_only(gui) ? 10f0 : Float32(gui.trainer.dataset.camera_extent)
            gui.control_settings.orbiting_target = view_pos(gui.camera) .+ d .* view_dir(gui.camera)
        end

        # Yaw-axis calibration: see `estimate_up_vec` & `level_horizon!`.
        CImGui.BeginTable("##up-vec-buttons-table", 2)
        CImGui.TableNextRow()
        CImGui.TableNextColumn()
        if CImGui.Button("Set Up From View", CImGui.ImVec2(-1, 0))
            gui.control_settings.up_vec = -view_up(gui.camera)
        end
        CImGui.SetItemTooltip(
            "Use the current camera up as the scene up: " *
            "yaw will rotate around it.")
        CImGui.TableNextColumn()
        if CImGui.Button("Level Horizon", CImGui.ImVec2(-1, 0))
            level_horizon!(gui.camera, gui.control_settings.up_vec)
            gui.render_state.need_render = true
        end
        CImGui.SetItemTooltip(
            "Remove accumulated roll: align the camera with the scene up.")
        CImGui.EndTable()

        has_dataset = !viewer_only(gui)
        has_dataset || disabled_begin()
        if CImGui.Button("Reset Up From Dataset", CImGui.ImVec2(-1, 0))
            gui.control_settings.up_vec =
                estimate_up_vec(gui.trainer.dataset.train_cameras)
        end
        CImGui.SetItemTooltip(
            "Re-estimate the scene up from the dataset cameras, " *
            "discarding manual calibration.")
        has_dataset || disabled_end()

        CImGui.PushItemWidth(-100)
        max_sh_degree = gui.ui_state.max_sh_degree
        if max_sh_degree > 0 && CImGui.SliderInt(
            "SH degree", gui.ui_state.sh_degree,
            -1, max_sh_degree, "%d / $max_sh_degree",
        )
            gui.render_state.need_render = true
        end

        # GUI rasterizers always render in `:rgbd` mode.
        CImGui.PushItemWidth(-100)
        if CImGui.Combo("Mode", gui.ui_state.selected_mode,
            gui.ui_state.render_modes,
        )
            gui.render_state.need_render = true
        end

        if !viewer_only(gui)
            CImGui.Separator()

            CImGui.BeginTable("##checkbox-table", 2)

            # Row 1.
            CImGui.TableNextRow()
            CImGui.TableNextColumn()
            CImGui.Text("Steps: $(w.step[])")
            CImGui.TableNextColumn()
            CImGui.Text("Loss: $(round(gui.ui_state.loss; digits=4))")

            # Row 2.
            CImGui.TableNextRow()
            CImGui.TableNextColumn()
            # Reflect worker-side stops (e.g. training error)
            # before drawing the checkbox.
            gui.ui_state.train[] = w.train[]
            if CImGui.Checkbox("Train", gui.ui_state.train)
                GC.gc(false)
                GC.gc(true)
                w.train[] = gui.ui_state.train[]
                notify(w.wakeup)
            end
            CImGui.TableNextColumn()
            CImGui.Checkbox("Draw Cameras", gui.ui_state.draw_cameras)

            # Row 3.
            CImGui.TableNextRow()
            CImGui.TableNextColumn()
            if CImGui.Checkbox("Densify", gui.ui_state.densify)
                w.densify[] = gui.ui_state.densify[]
            end
            CImGui.TableNextColumn()

            CImGui.EndTable()

            if gui.ui_state.is_mcmc
                # Benign cross-thread write: `max_cap` is a
                # word-sized Int the worker only reads at
                # densification time.
                strategy = gui.trainer.strategy
                max_cap_ref = Ref{Int32}(strategy.max_cap)
                CImGui.PushItemWidth(-100)
                if CImGui.InputInt("Max Gaussians", max_cap_ref, 100_000, 500_000)
                    strategy.max_cap = max(w.n_gaussians[], Int(max_cap_ref[]))
                end
            end

            image_filenames = gui.trainer.dataset.train_image_filenames
            CImGui.Text("Camera view:")
            CImGui.PushItemWidth(-1)
            if CImGui.ListBox("##views", gui.ui_state.selected_view,
                image_filenames, VIEW_LIST_ROWS,
            )
                vid = gui.ui_state.selected_view[] + 1
                set_c2w!(gui.camera, gui.trainer.dataset.train_cameras[vid].c2w)
                # A dataset photo's pose is level: use it to calibrate the yaw axis.
                gui.control_settings.up_vec = -view_up(gui.camera)
                gui.render_state.need_render = true
            end
        end
    end
    CImGui.End()
    return
end

function render!(gui::GSGUI)
    gui.render_state.need_render || return

    # `need_render` is `true` every time user interacts with the app
    # via controls, so we need to render anew.
    if gui.render_state.need_render
        gui.render_state.need_render = false
    end

    gs = gui.gaussians
    rast = gui.rasterizer
    (; width, height) = resolution(gui.camera)

    # Empty scene (no dataset loaded yet): display background color.
    if gs ≡ nothing || length(gs) == 0
        NGL.set_data!(gui.render_state.surface,
            zeros(Float32, 3, width, height))
        return
    end

    # The worker normally rebuilds the rasterizer on resolution change;
    # this synchronous path (capture mode) has to do it itself.
    if size(rast.image)[2:3] != (width, height)
        kab = get_backend(rast)
        # TODO free the old one before creating new one.
        gui.rasterizer = rast =
            GaussianRasterizer(kab, gui.camera; mode=:rgbd)
    end

    ui_sh_degree::Int = gui.ui_state.sh_degree[]
    sh_degree = ui_sh_degree == -1 ? gs.sh_degree : ui_sh_degree
    rast(
        gs.points, gs.opacities, gs.scales,
        gs.rotations, gs.features_dc, gs.features_rest;
        camera=gui.camera, sh_degree)

    mode = gui.ui_state.selected_mode[]
    tex = if mode == 0 # Render color.
        gl_texture(rast)
    elseif mode == 1 # Render depth.
        gl_depth(rast)
    end
    NGL.set_data!(gui.render_state.surface, tex)
    return
end
