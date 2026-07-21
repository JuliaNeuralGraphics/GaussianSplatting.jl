# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
include("ui_state.jl")
include("render_state.jl")
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
    # `nothing` on empty startup (`app`), until a dataset/model is loaded.
    gaussians::Maybe{GaussianModel}
    rasterizer::GaussianRasterizer
    trainer::Maybe{Trainer}
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
    kab = get_backend(gui.rasterizer)
    # TODO free the old one before creating new one.
    gui.rasterizer = GaussianRasterizer(kab, gui.camera; mode=gui.rasterizer.mode)
    gui.render_state.need_render = true
    return
end

# Viewer-only mode.
function GSGUI(kab, gaussians::Maybe{GaussianModel}, camera::Camera; gl_kwargs...)
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
    rasterizer = GaussianRasterizer(kab, camera; fused=true)

    render_state = RenderState(;
        surface=NGL.RenderSurface(;
            internal_format=GL_RGB32F, data_type=GL_FLOAT,
            resolution(camera)...),
        framebuffer=NGL.Framebuffer(; resolution(camera)...))
    control_settings = ControlSettings()
    ui_state = UIState()

    capture_mode = CaptureMode()

    trainer = nothing
    gsgui = GSGUI(
        context, MainScreen, NGL.Frustum(), render_state, ui_state,
        control_settings, capture_mode, camera,
        gaussians, rasterizer, trainer)
    GSGUI_REF[] = gsgui
    return gsgui
end

# Densification strategies selectable in the UI.
const STRATEGIES = (:default, :mcmc)

# Training mode.
function GSGUI(kab, dataset_path::String, scale::Int;
    strategy::Symbol = :default, gl_kwargs...,
)
    NGL.init(3, 2)
    context = NGL.Context("GaussianSplatting.jl"; gl_kwargs...)
    NGL.set_resize_callback!(context, resize_callback)

    font_file = joinpath(pkgdir(CImGui), "fonts", "Roboto-Medium.ttf")
    fonts = unsafe_load(CImGui.GetIO().Fonts)
    CImGui.AddFontFromFileTTF(fonts, font_file, 16)

    enable_docking!()

    dataset = ColmapDataset(kab, dataset_path; scale, train_test_split=1)
    camera = dataset.train_cameras[1]

    opt_params = OptimizationParams()
    gaussians = GaussianModel(dataset.points, dataset.colors, dataset.scales;
        isotropic=false, max_sh_degree=3)
    rasterizer = GaussianRasterizer(kab, camera; fused=true)
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params;
        strategy=create_strategy(strategy, gaussians))

    # Set-up separate renderer camera & rasterizer.
    camera = deepcopy(camera)
    set_resolution!(camera; (;
        width=16 * cld(context.width, 16),
        height=16 * cld(context.height, 16))...)
    gui_rasterizer = GaussianRasterizer(kab, camera; fused=true, mode=:rgbd)

    render_state = RenderState(;
        surface=NGL.RenderSurface(;
            internal_format=GL_RGB32F, data_type=GL_FLOAT,
            resolution(camera)...),
        framebuffer=NGL.Framebuffer(; resolution(camera)...))
    control_settings = ControlSettings()
    control_settings.up_vec = estimate_up_vec(dataset.train_cameras)
    ui_state = UIState()

    capture_mode = CaptureMode()

    gsgui = GSGUI(
        context, MainScreen, NGL.Frustum(), render_state, ui_state,
        control_settings, capture_mode, camera,
        gaussians, gui_rasterizer, trainer)
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
)
    dataset = ColmapDataset(kab, dataset_path; scale, train_test_split=1)
    camera = dataset.train_cameras[1]

    opt_params = OptimizationParams()
    gaussians = GaussianModel(dataset.points, dataset.colors, dataset.scales;
        isotropic=false, max_sh_degree=3)
    rasterizer = GaussianRasterizer(kab, camera; fused=true)
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params;
        strategy=create_strategy(strategy, gaussians))

    # Set-up separate renderer camera & rasterizer.
    camera = deepcopy(camera)
    set_resolution!(camera; width, height)
    # TODO free the old one before creating new one.
    gui_rasterizer = GaussianRasterizer(kab, camera; fused=true, mode=:rgbd)

    up_vec = estimate_up_vec(dataset.train_cameras)
    return (; camera, gaussians, gui_rasterizer, trainer, up_vec)
end

# Replace the current scene, keeping the GL context & render surface.
function apply_dataset!(gui::GSGUI, loaded)
    gui.camera = loaded.camera
    gui.gaussians = loaded.gaussians
    gui.rasterizer = loaded.gui_rasterizer
    gui.trainer = loaded.trainer
    # Yaw rotates around the estimated scene up: keeps the horizon level.
    gui.control_settings.up_vec = loaded.up_vec

    reset_ui!(gui.ui_state)
    gui.render_state.need_render = true
    return
end

"""
Load a `.bson` model checkpoint at runtime in viewer-only mode
(no trainer), replacing the current scene.
"""
function load_bson!(gui::GSGUI, state_file::String)
    kab = get_backend(gui.rasterizer)

    θ = BSON.load(state_file)
    gaussians = GaussianModel(kab)
    set_from_bson!(gaussians, θ[:gaussians])

    # Keep the current render resolution.
    camera = θ[:camera]
    set_resolution!(camera; resolution(gui.camera)...)
    # TODO free the old one before creating new one.
    gui_rasterizer = GaussianRasterizer(kab, camera; fused=true, mode=:rgbd)

    gui.camera = camera
    gui.gaussians = gaussians
    gui.rasterizer = gui_rasterizer
    gui.trainer = nothing

    reset_ui!(gui.ui_state)
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
                try
                    load_bson!(gui, state_file)
                catch err
                    @error "Failed to load BSON checkpoint from `$state_file`:" exception=(err, catch_backtrace())
                end
            end
        end

        CImGui.Separator()

        # Saving needs a trainer: it stores optimizers & training step.
        if CImGui.MenuItem("Save BSON...", C_NULL, false, !viewer_only(gui))
            state_file = save_file(homedir(); filterlist="bson") # Empty when cancelled.
            if !isempty(state_file)
                endswith(state_file, ".bson") || (state_file *= ".bson")
                try
                    save_state(gui.trainer, state_file)
                catch err
                    @error "Failed to save BSON checkpoint to `$state_file`:" exception=(err, catch_backtrace())
                end
            end
        end
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
            CImGui.Text("Loading dataset. Please wait...")
            # Negative fraction switches ProgressBar into indeterminate mode.
            CImGui.ProgressBar(
                -1f0 * Float32(CImGui.GetTime()),
                CImGui.ImVec2(-1f0, 0f0), "Loading...")
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
        (; width, height) = resolution(gui.camera)
        ui_state.dataset_load_task = Threads.@spawn load_dataset(
            kab, dataset_path; scale, width, height, strategy)
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
    NGL.render_loop(gui.context) do
        if gui.screen == MainScreen
            loop!(gui)
        else gui.screen == CaptureScreen
            loop!(gui.capture_mode; gui)
        end
        return true
    end
end

function loop!(gui::GSGUI)
    frame_time = update_time!(gui.render_state)
    NGL.imgui_begin()
    menu_bar!(gui)
    open_dataset_modal!(gui)
    dockspace_id = dockspace!()

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

    do_train = gui.ui_state.train[] && !mouse_in_ui
    if do_train
        gui.ui_state.loss = step!(gui.trainer)
        gui.render_state.need_render = true
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
    return
end

"""
Pick a new orbiting target by unprojecting the rendered depth under
the `(px, py)` pixel (1-based, top-left origin).
Depth is averaged over a small window around the pixel to be robust
to outliers at fuzzy silhouettes; background pixels (nothing rendered
there, so the blended depth is ≈ 0) are excluded.
Return `nothing` when the click hit only background.
"""
function pick_orbit_target(
    gui::GSGUI, px::Integer, py::Integer,
)::Maybe{SVector{3, Float32}}
    gs = gui.gaussians
    (gs ≡ nothing || length(gs) == 0) && return nothing

    rast = gui.rasterizer
    rast.mode == :rgbd || return nothing

    (; width, height) = resolution(gui.camera)
    (1 ≤ px ≤ width && 1 ≤ py ≤ height) || return nothing

    δ = 4 # Window half-size.
    depths = Array(@view(rast.image[4,
        max(1, px - δ):min(width, px + δ),
        max(1, py - δ):min(height, py + δ)]))
    valid = depths .> 1f-2 # Near plane: rejects background pixels.
    any(valid) || return nothing
    z = mean(depths[valid])

    # Unproject through the camera intrinsics
    # (COLMAP frame: x right, y down, z forward).
    fx, fy = gui.camera.intrinsics.focal
    cx = gui.camera.intrinsics.principal[1] * width
    cy = gui.camera.intrinsics.principal[2] * height
    p_cam = SVector{3, Float32}(
        (px - 0.5f0 - cx) * z / fx,
        (py - 0.5f0 - cy) * z / fy,
        z)
    R = SMatrix{3, 3, Float32}(@view(gui.camera.c2w[1:3, 1:3]))
    return R * p_cam .+ view_pos(gui.camera)
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

    (force_render || gui.ui_state.render[]) && render!(gui)
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

        # Double-click in orbiting mode: set the orbiting target to the
        # point under the cursor. The image is drawn 1:1, so the offset
        # from its top-left corner is the pixel position.
        if gui.ui_state.controller_mode[] == 1 &&
            CImGui.IsItemHovered() && CImGui.IsMouseDoubleClicked(0)

            rect_min = CImGui.GetItemRectMin()
            mouse_pos = CImGui.GetMousePos()
            target = pick_orbit_target(gui,
                floor(Int, mouse_pos.x - rect_min.x) + 1,
                floor(Int, mouse_pos.y - rect_min.y) + 1)
            target ≢ nothing &&
                (gui.control_settings.orbiting_target = target)
        end
    end
    CImGui.End()

    gui.ui_state.scene_hovered = hovered
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
    if CImGui.Begin("GaussianSplatting")
        if CImGui.BeginTabBar("bar")
            if CImGui.BeginTabItem("Controls")
                (; width, height) = resolution(gui.camera)
                CImGui.Text("Render Resolution: $width x $height")
                n_gaussians = gui.gaussians ≡ nothing ? 0 : length(gui.gaussians)
                CImGui.Text("N Gaussians: $n_gaussians")

                CImGui.Checkbox("Render", gui.ui_state.render)

                CImGui.PushItemWidth(-100)
                if CImGui.Combo("Controller", gui.ui_state.controller_mode,
                    gui.ui_state.controller_modes, length(gui.ui_state.controller_modes),
                ) && gui.ui_state.controller_mode[] == 1
                    # Entering orbit mode: place the target in front of the
                    # camera, at a scene-sized distance.
                    d = viewer_only(gui) ?
                        10f0 : Float32(gui.trainer.dataset.camera_extent)
                    gui.control_settings.orbiting_target =
                        view_pos(gui.camera) .+ d .* view_dir(gui.camera)
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
                max_sh_degree = gui.gaussians ≡ nothing ?
                    0 : gui.gaussians.max_sh_degree
                if max_sh_degree > 0 && CImGui.SliderInt(
                    "SH degree", gui.ui_state.sh_degree,
                    -1, max_sh_degree, "%d / $max_sh_degree",
                )
                    gui.render_state.need_render = true
                end

                if gui.rasterizer.mode == :rgbd
                    CImGui.PushItemWidth(-100)
                    if CImGui.Combo("Mode", gui.ui_state.selected_mode,
                        gui.ui_state.render_modes, length(gui.ui_state.render_modes),
                    )
                        gui.render_state.need_render = true
                    end
                end

                if !viewer_only(gui)
                    CImGui.Separator()

                    CImGui.BeginTable("##checkbox-table", 2)

                    # Row 1.
                    CImGui.TableNextRow()
                    CImGui.TableNextColumn()
                    CImGui.Text("Steps: $(gui.trainer.step)")
                    CImGui.TableNextColumn()
                    CImGui.Text("Loss: $(round(gui.ui_state.loss; digits=4))")

                    # Row 2.
                    CImGui.TableNextRow()
                    CImGui.TableNextColumn()
                    if CImGui.Checkbox("Train", gui.ui_state.train)
                        GC.gc(false)
                        GC.gc(true)
                    end
                    CImGui.TableNextColumn()
                    CImGui.Checkbox("Draw Cameras", gui.ui_state.draw_cameras)

                    # Row 3.
                    CImGui.TableNextRow()
                    CImGui.TableNextColumn()
                    if CImGui.Checkbox("Densify", gui.ui_state.densify)
                        gui.trainer.densify = gui.ui_state.densify[]
                    end
                    CImGui.TableNextColumn()

                    CImGui.EndTable()

                    image_filenames = gui.trainer.dataset.train_image_filenames
                    CImGui.Text("Camera view:")
                    CImGui.PushItemWidth(-1)
                    if CImGui.ListBox("##views", gui.ui_state.selected_view,
                        image_filenames, length(image_filenames),
                    )
                        vid = gui.ui_state.selected_view[] + 1
                        set_c2w!(gui.camera, gui.trainer.dataset.train_cameras[vid].c2w)
                        # A dataset photo's pose is level: use it to calibrate the yaw axis.
                        gui.control_settings.up_vec = -view_up(gui.camera)
                        gui.render_state.need_render = true
                    end
                end
                CImGui.EndTabItem()
            end

            if CImGui.BeginTabItem("Utils")
                if CImGui.Button("Capture Video", CImGui.ImVec2(-1, 0))
                    GC.gc(false)
                    GC.gc(true)
                    gui.screen = CaptureScreen
                    NGL.set_resizable_window!(gui.context, false)
                end
                CImGui.EndTabItem()
            end

            if CImGui.BeginTabItem("Help")
                CImGui.TextWrapped("FPV controller:")
                CImGui.TextWrapped("- Left Mouse to rotate camera.")
                CImGui.TextWrapped("- WASD to move the camera.")
                CImGui.TextWrapped("- QE to move the camera up/down.")
                CImGui.TextWrapped("- R + Left Mouse to control the roll.")
                CImGui.TextWrapped(" ")

                CImGui.TextWrapped("Orbiting controller:")
                CImGui.TextWrapped("- Left Mouse to pan (moves the orbiting target).")
                CImGui.TextWrapped("- Right Mouse to orbit around the target.")
                CImGui.TextWrapped("- Mouse Wheel to zoom towards the target.")
                CImGui.TextWrapped("- Double Left Click to pick a new orbiting target.")
                CImGui.EndTabItem()
            end
            CImGui.EndTabBar()
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

    # Empty scene (no dataset loaded yet): display background color.
    if gs ≡ nothing || length(gs) == 0
        (; width, height) = resolution(gui.camera)
        NGL.set_data!(gui.render_state.surface,
            zeros(Float32, 3, width, height))
        return
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
