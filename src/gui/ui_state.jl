const DEFAULT_MAX_STEPS = Int32(30_000)
const DEFAULT_AUTOSAVE_EVERY = Int32(10_000)

Base.@kwdef mutable struct UIState
    train::Ref{Bool} = Ref(false)
    render::Ref{Bool} = Ref(true)
    densify::Ref{Bool} = Ref(true)
    # Training stops itself at this step; `0` means no limit.
    max_steps::Ref{Int32} = Ref(DEFAULT_MAX_STEPS)
    # Periodic checkpointing (see `maybe_autosave!`): every `autosave_every`
    # steps & on the run's final step. `0` steps saves only the final one.
    autosave::Ref{Bool} = Ref(false)
    autosave_every::Ref{Int32} = Ref(DEFAULT_AUTOSAVE_EVERY)
    # Base path the saved filenames are derived from; empty until one is
    # picked, which is also what keeps `autosave` from being switched on.
    autosave_path::String = ""
    scene_hovered::Bool = false

    loss::Float32 = 0f0
    loss_ema::Float32 = 0f0

    # Dataset camera overlay (see `draw_dataset_frustums!`).
    draw_cameras::Ref{Bool} = Ref(false)
    draw_camera_images::Ref{Bool} = Ref(true)
    # Frustum size, relative to the scene extent (see `frustum_scale`).
    camera_size::Ref{Float32} = Ref(1f0)
    camera_image_opacity::Ref{Float32} = Ref(1f0)
    selected_view::Ref{Int32} = Ref{Int32}(0)

    sh_degree::Ref{Int32} = Ref{Int32}(-1) # -1 means use value from the model

    selected_mode::Ref{Int32} = Ref{Int32}(0)
    render_modes::Vector{String} = ["Color", "Depth"]

    # Color the splats are composited over in the viewer (RGB in [0, 1]).
    # A `Vector` because ImGui's color picker writes through the pointer.
    background_color::Vector{Float32} = zeros(Float32, 3)

    # UI-side caches of worker-owned state (set on scene install).
    max_sh_degree::Int = 0
    # Whether the active strategy exposes a `max_cap` knob.
    has_max_cap::Bool = false
    strategy_name::String = ""
    # Last error reported by the render worker; empty when none.
    worker_error::String = ""

    controller_mode::Ref{Int32} = Ref{Int32}(0)
    controller_modes::Vector{String} = ["FPV", "Orbiting"]

    # Whether the `Capture` tab is the open one: capture mode is active
    # (camera path drawn & editable). Refreshed every frame by `handle_ui!`.
    capture_tab::Bool = false

    # `Open Dataset` modal.
    open_dataset_popup::Bool = false
    dataset_scale::Ref{Int32} = Ref{Int32}(1)
    # Index into `STRATEGIES` (densification strategy for the new trainer),
    # `0` being its first entry: the default strategy.
    dataset_strategy::Ref{Int32} = Ref{Int32}(0)
    # Highest SH band the new model allocates (see `GaussianModel`).
    dataset_max_sh_degree::Ref{Int32} = Ref{Int32}(3)
    # Per-train-image appearance modeling (see `bilateral_grid.jl`).
    dataset_bilateral_grid::Ref{Bool} = Ref(false)
    # Monocular depth supervision (see `depth_supervision.jl`).
    dataset_depth_loss::Ref{Bool} = Ref(true)
    # Geometry regularization (see `geometry_regularization.jl`).
    dataset_normal_loss::Ref{Bool} = Ref(false)
    # Far-field shell the sky is painted onto (see `sky_dome.jl`).
    dataset_sky_dome::Ref{Bool} = Ref(false)
    # Index into `SKY_DOME_SHAPES`.
    dataset_sky_dome_shape::Ref{Int32} = Ref{Int32}(0)
    # Supervise the accumulated alpha with the dataset's `sky/` masks
    # (see `sky_dome.jl`). Inert unless the dataset has them.
    dataset_sky_loss::Ref{Bool} = Ref(false)
    # Composite train renders over a random background (see `OptimizationParams`).
    dataset_random_background::Ref{Bool} = Ref(false)
    # Read the dataset's `masks/` directory (see `masking.jl`). Gates the
    # dataset side of it too, not just the loss: `ColmapDataset(; use_masks)`.
    dataset_masks::Ref{Bool} = Ref(true)
    # Skip the Adam update for gaussians not visible this step (see `sparse_adam.jl`).
    dataset_sparse_adam::Ref{Bool} = Ref(false)
    # Hyperparameters the new trainer starts from: defaults, or everything a
    # loaded TOML file specifies (see `params_io.jl`). The controls above own
    # the fields they expose & are applied over this on `Open`.
    dataset_opt_params::OptimizationParams = OptimizationParams()
    # Where those came from; empty means untouched defaults.
    dataset_params_file::String = ""
    dataset_path::Vector{UInt8} = Vector{UInt8}("\0"^1024)
    dataset_error::String = ""
    dataset_load_task::Maybe{Task} = nothing

    # `Open Checkpoint` / `Open PLY` model loading
    # (see `load_checkpoint_model`, `load_ply` & `poll_model_load!`).
    model_load_task::Maybe{Task} = nothing
end
