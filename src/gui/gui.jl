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

mutable struct GSGUI{
    T <: Trainer,
    R <: GaussianRasterizer,
}
    context::NGL.Context
    screen::Screen
    frustum::NGL.Frustum
    render_state::RenderState
    ui_state::UIState
    control_settings::ControlSettings

    capture_mode::CaptureMode

    camera::Camera
    rasterizer::R
    trainer::T
end

const GSGUI_REF::Ref{GSGUI} = Ref{GSGUI}()

function resize_callback(_, width, height)
    (width == 0 || height == 0) && return # Window minimized.

    NGL.set_viewport(width, height)
    isassigned(GSGUI_REF) || return

    width, height = 16 * cld(width, 16), 16 * cld(height, 16)

    gsgui::GSGUI = GSGUI_REF[]
    NGL.resize!(gsgui.render_state.surface; width, height)

    set_resolution!(gsgui.camera; width, height)
    kab = get_backend(gsgui.rasterizer)
    # TODO free the old one before creating new one.
    gsgui.rasterizer = GaussianRasterizer(kab, gsgui.camera)
    gsgui.render_state.need_render = true
    return
end

function GSGUI(dataset_path::String, scale::Int; gl_kwargs...)
    kab = Backend
    get_module(kab).allowscalar(false)

    NGL.init(3, 0)
    context = NGL.Context("GaussianSplatting.jl"; gl_kwargs...)
    NeuralGraphicsGL.set_resize_callback!(context, resize_callback)

    font_file = joinpath(pkgdir(CImGui), "fonts", "Roboto-Medium.ttf")
    fonts = unsafe_load(CImGui.GetIO().Fonts)
    CImGui.AddFontFromFileTTF(fonts, font_file, 16)

    cameras_file = joinpath(dataset_path, "sparse/0/cameras.bin")
    images_file = joinpath(dataset_path, "sparse/0/images.bin")
    points_file = joinpath(dataset_path, "sparse/0/points3D.bin")
    images_dir = joinpath(dataset_path, "images")
    dataset = ColmapDataset(kab;
        cameras_file, images_file, points_file, images_dir, scale)

    opt_params = OptimizationParams()
    gaussians = GaussianModel(dataset.points, dataset.colors, dataset.scales)
    rasterizer = GaussianRasterizer(kab, dataset.cameras[1])
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params)

    # Set-up separate renderer camera & rasterizer.
    camera = deepcopy(dataset.cameras[1])
    render_resolution = (;
        width=16 * cld(context.width, 16),
        height=16 * cld(context.height, 16))
    set_resolution!(camera; render_resolution...)
    gui_rasterizer = GaussianRasterizer(kab, camera)

    # TODO show camera resolution in UI
    @show resolution(camera)
    render_state = RenderState(; surface=NGL.RenderSurface(;
        internal_format=GL_RGB32F, data_type=GL_FLOAT,
        resolution(camera)...))
    control_settings = ControlSettings()
    ui_state = UIState()

    capture_mode = CaptureMode()

    gsgui = GSGUI(
        context, MainScreen, NGL.Frustum(), render_state, ui_state,
        control_settings, capture_mode, camera, gui_rasterizer, trainer)
    GSGUI_REF[] = gsgui
    return gsgui
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
    NGL.imgui_begin(gui.context)

    # Handle controls.

    handle_ui!(gui; frame_time)
    gui.render_state.need_render |= handle_keyboard!(
        gui.control_settings, gui.camera; frame_time)
    gui.render_state.need_render |= handle_mouse!(
        gui.control_settings, gui.camera)

    do_train = gui.ui_state.train[] && !is_mouse_in_ui()
    if do_train
        gui.ui_state.loss = step!(gui.trainer)
        gui.render_state.need_render = true
    end

    # Draw gaussians.

    NGL.clear()
    NGL.set_clear_color(0.2, 0.2, 0.2, 1.0)
    if gui.ui_state.render[]
        render!(gui)
    end
    NGL.draw(gui.render_state.surface)
    NGL.clear(NGL.GL_DEPTH_BUFFER_BIT)

    # Draw other OpenGL objects.

    P = NeuralGraphicsGL.perspective(gui.camera)
    L = NeuralGraphicsGL.look_at(gui.camera)

    if gui.ui_state.draw_cameras[]
        dataset = gui.trainer.dataset
        for view_id in 1:length(dataset)
            camera = dataset.cameras[view_id]
            camera_perspective =
                NGL.perspective(camera; near=0.1f0, far=0.2f0) *
                NGL.look_at(camera)
            NGL.draw(gui.frustum, camera_perspective, P, L)
        end
    end

    NGL.imgui_end(gui.context)
    glfwSwapBuffers(gui.context.window)
    glfwPollEvents()
    return
end

function handle_ui!(gui::GSGUI; frame_time)
    CImGui.Begin("GaussianSplatting")

    (; width, height) = resolution(gui.camera)
    CImGui.Text("Render Resolution: $width x $height")
    CImGui.Text("N Gaussians: $(size(gui.trainer.gaussians.points, 2))")
    CImGui.Text("Steps: $(gui.trainer.step)")
    CImGui.Text("Loss: $(round(gui.ui_state.loss; digits=6))")
    if CImGui.Checkbox("Train", gui.ui_state.train)
        GC.gc(false)
        GC.gc(true)
    end
    CImGui.Checkbox("Render", gui.ui_state.render)
    CImGui.Checkbox("Draw Cameras", gui.ui_state.draw_cameras)

    image_filenames = gui.trainer.dataset.image_filenames
    CImGui.PushItemWidth(-100)
    if CImGui.Combo("View", gui.ui_state.selected_view,
        image_filenames, length(image_filenames),
    )
        vid = gui.ui_state.selected_view[] + 1
        set_c2w!(gui.camera, gui.trainer.dataset.cameras[vid].c2w)
        gui.render_state.need_render = true
    end

    CImGui.Text("Path to save directory:")
    CImGui.PushItemWidth(-1)
    CImGui.InputText(
        "##savedir-inputtext", pointer(gui.ui_state.save_directory_path),
        length(gui.ui_state.save_directory_path))

    if CImGui.Button("Save", CImGui.ImVec2(-1, 0))
        save_dir = unsafe_string(pointer(gui.ui_state.save_directory_path))
        save_file = joinpath(save_dir, "gsp-$(gui.trainer.step).bson")
        save_state(gui.trainer, save_file)
    end

    CImGui.Text("Path to state file:")
    CImGui.PushItemWidth(-1)
    CImGui.InputText(
        "##statefile-inputtext", pointer(gui.ui_state.state_file),
        length(gui.ui_state.state_file))

    if CImGui.Button("Load", CImGui.ImVec2(-1, 0))
        state_file = unsafe_string(pointer(gui.ui_state.state_file))
        if isfile(state_file)
            load_state!(gui.trainer, state_file)
        end
    end

    if CImGui.Button("Capture Mode", CImGui.ImVec2(-1, 0))
        GC.gc(false)
        GC.gc(true)
        gui.screen = CaptureScreen
        NGL.set_resizable_window!(gui.context, false)
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

    gs = gui.trainer.gaussians
    rast = gui.rasterizer

    shs = hcat(gs.features_dc, gs.features_rest)
    rast(
        gs.points, gs.opacities, gs.scales,
        gs.rotations, shs; camera=gui.camera, sh_degree=gs.sh_degree)

    tex = gl_texture(rast)
    NGL.set_data!(gui.render_state.surface, tex)
    return
end
