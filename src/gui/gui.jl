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
    G <: GaussianModel,
    T <: Maybe{Trainer},
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
    gaussians::G
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
    gsgui.rasterizer = GaussianRasterizer(kab, gsgui.camera; mode=gsgui.rasterizer.mode)
    gsgui.render_state.need_render = true
    return
end

# Viewer-only mode.
function GSGUI(gaussians::GaussianModel, camera::Camera; gl_kwargs...)
    kab = gpu_backend()

    NGL.init(3, 0)
    context = NGL.Context("GaussianSplatting.jl"; gl_kwargs...)
    NGL.set_resize_callback!(context, resize_callback)

    font_file = joinpath(pkgdir(CImGui), "fonts", "Roboto-Medium.ttf")
    fonts = unsafe_load(CImGui.GetIO().Fonts)
    CImGui.AddFontFromFileTTF(fonts, font_file, 16)

    # Set up renderer.
    set_resolution!(camera; (;
        width=16 * cld(context.width, 16),
        height=16 * cld(context.height, 16))...)
    rasterizer = GaussianRasterizer(kab, camera; fused=true)

    render_state = RenderState(; surface=NGL.RenderSurface(;
        internal_format=GL_RGB32F, data_type=GL_FLOAT,
        resolution(camera)...))
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

# Training mode.
function GSGUI(dataset_path::String, scale::Int; gl_kwargs...)
    kab = gpu_backend()

    NGL.init(3, 0)
    context = NGL.Context("GaussianSplatting.jl"; gl_kwargs...)
    NGL.set_resize_callback!(context, resize_callback)

    font_file = joinpath(pkgdir(CImGui), "fonts", "Roboto-Medium.ttf")
    fonts = unsafe_load(CImGui.GetIO().Fonts)
    CImGui.AddFontFromFileTTF(fonts, font_file, 16)

    dataset = ColmapDataset(kab, dataset_path; scale, train_test_split=1)
    camera = dataset.train_cameras[1]

    opt_params = OptimizationParams()
    gaussians = GaussianModel(dataset.points, dataset.colors, dataset.scales)
    rasterizer = GaussianRasterizer(kab, camera; fused=true)
    trainer = Trainer(rasterizer, gaussians, dataset, opt_params)

    # Set-up separate renderer camera & rasterizer.
    camera = deepcopy(camera)
    set_resolution!(camera; (;
        width=16 * cld(context.width, 16),
        height=16 * cld(context.height, 16))...)
    gui_rasterizer = GaussianRasterizer(kab, camera; fused=true, mode=:rgb)

    render_state = RenderState(; surface=NGL.RenderSurface(;
        internal_format=GL_RGB32F, data_type=GL_FLOAT,
        resolution(camera)...))
    control_settings = ControlSettings()
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

    # Handle controls.
    mouse_in_ui = is_mouse_in_ui()

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

    if !viewer_only(gui) && gui.ui_state.draw_cameras[]
        dataset = gui.trainer.dataset
        for view_id in 1:length(dataset)
            camera = dataset.train_cameras[view_id]
            camera_perspective =
                NGL.perspective(camera; near=0.1f0, far=0.2f0) *
                NGL.look_at(camera)
            NGL.draw(gui.frustum, camera_perspective, P, L)
        end
    end

    NGL.imgui_end()
    GLFW.SwapBuffers(gui.context.window)
    GLFW.PollEvents()
    return
end

function handle_ui!(gui::GSGUI; frame_time)
    if CImGui.Begin("GaussianSplatting")
        if CImGui.BeginTabBar("bar")
            if CImGui.BeginTabItem("Controls")
                (; width, height) = resolution(gui.camera)
                CImGui.Text("Render Resolution: $width x $height")
                CImGui.Text("N Gaussians: $(size(gui.gaussians.points, 2))")

                CImGui.Checkbox("Render", gui.ui_state.render)

                CImGui.PushItemWidth(-100)
                CImGui.Combo("Controller", gui.ui_state.controller_mode,
                    gui.ui_state.controller_modes, length(gui.ui_state.controller_modes),
                )

                CImGui.PushItemWidth(-100)
                max_sh_degree = gui.gaussians.sh_degree
                if CImGui.SliderInt(
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
                        gui.render_state.need_render = true
                    end
                end
                CImGui.EndTabItem()
            end

            if CImGui.BeginTabItem("Save/Load")
                CImGui.Text("Path to Save Directory:")

                CImGui.PushItemWidth(-1)
                CImGui.InputText(
                    "##savedir-inputtext", pointer(gui.ui_state.save_directory_path),
                    length(gui.ui_state.save_directory_path))

                if CImGui.Button("Save", CImGui.ImVec2(-1, 0))
                    save_dir = unsafe_string(pointer(gui.ui_state.save_directory_path))
                    isdir(save_dir) || mkpath(save_dir)

                    tstmp = now()
                    fmt = "timestamp-$(month(tstmp))-$(day(tstmp))-$(hour(tstmp)):$(minute(tstmp))"
                    save_file = joinpath(save_dir, "state-(step-$(gui.trainer.step))-($fmt).bson")
                    save_state(gui.trainer, save_file)
                end
                CImGui.Separator()

                CImGui.Text("Path to State File (.bson):")
                CImGui.PushItemWidth(-1)
                CImGui.InputText(
                    "##statefile-inputtext", pointer(gui.ui_state.state_file),
                    length(gui.ui_state.state_file))

                if CImGui.Button("Load", CImGui.ImVec2(-1, 0))
                    state_file = unsafe_string(pointer(gui.ui_state.state_file))
                    if isfile(state_file)
                        if endswith(state_file, ".bson")
                            load_state!(gui.trainer, state_file)
                            gui.render_state.need_render = true
                        else
                            gui.ui_state.state_file = Vector{UInt8}(
                                "<invalid-file-extension>" * "\0"^512)
                        end
                    else
                        gui.ui_state.state_file = Vector{UInt8}(
                            "<file-does-not-exist>" * "\0"^512)
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
                CImGui.TextWrapped("- Left Mouse to shift camera sideways and forward/backward.")
                CImGui.TextWrapped("- F + Left Mouse to shift camera sideways and up/down.")
                CImGui.TextWrapped("- Right Mouse to orbit camera.")
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
