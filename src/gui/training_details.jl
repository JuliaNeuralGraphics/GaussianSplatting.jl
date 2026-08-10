# The `Training Details` window: every hyperparameter the running trainer uses.
#
# Its own dockable window rather than a tab of the controls window: the values
# are what you read *while* watching the scene react to them, so it has to be
# possible to keep it open next to the view — or to pull it out onto a second
# monitor entirely.

"""
`OptimizationParams` grouped the way the training code consumes it,
each group paired with the flag that switches it off (`nothing` when it always runs).
"""
const OPT_PARAM_SECTIONS = (
    (title="Photometric", flag=nothing, fields=(
        :λ_dssim, :random_background)),
    (title="Learning rates", flag=nothing, fields=(
        :lr_points_start, :lr_points_end, :lr_points_steps,
        :lr_feature, :lr_opacities, :lr_scales, :lr_rotations)),
    (title="Depth supervision", flag=:use_depth_loss, fields=(
        :depth_loss_weight, :depth_loss_mode, :depth_loss_steps,
        :depth_loss_final_scale, :depth_loss_gradient_weight)),
    (title="Sky dome", flag=:use_sky_dome, fields=(
        :sky_dome_shape, :sky_dome_points, :sky_dome_radius_factor,
        :sky_dome_lr)),
    (title="Sky mask supervision", flag=:use_sky_loss, fields=(
        :sky_loss_weight, :sky_loss_from_iter)),
    (title="Coverage masks", flag=:use_masks, fields=(
        :mask_opacity_weight, :mask_opacity_from_iter)),
    (title="Bilateral grid", flag=:use_bilateral_grid, fields=(
        :bilateral_grid_size, :bilateral_grid_lr,
        :bilateral_grid_lr_steps, :tv_loss_weight)),
    (title="Geometry regularization", flag=:use_normal_loss, fields=(
        :normal_consistency_weight, :normal_flatten_weight,
        :normal_from_iter)),
)

# A field added to `OptimizationParams` but not to a section would silently stop
# being reported, which is exactly what this panel exists to prevent: fail at
# load time instead.
let listed = Set(Iterators.flatten(
        section.flag ≡ nothing ? section.fields :
            (section.flag, section.fields...)
        for section in OPT_PARAM_SECTIONS)),
    known = Set(fieldnames(OptimizationParams))

    listed == known || error(
        "`OPT_PARAM_SECTIONS` must list every `OptimizationParams` field. " *
        "Missing: $(join(setdiff(known, listed), ", ")). " *
        "Unknown: $(join(setdiff(listed, known), ", ")).")
end

format_param(x) = string(x)
format_param(x::Symbol) = ":$x"
format_param(x::Tuple) = join(x, " × ")

"""
`seconds` as `1h 23m 45s` / `23m 45s` / `45.6s`.

Coarse units drop the decimal: at that length a tenth of a second is noise, and
the line is watched for how far a run has come, not measured with.
"""
function format_duration(seconds::Real)
    seconds < 60 && return "$(round(seconds; digits=1))s"

    minutes, secs = divrem(round(Int, seconds), 60)
    hours, mins = divrem(minutes, 60)
    return hours > 0 ? "$(hours)h $(mins)m $(secs)s" : "$(mins)m $(secs)s"
end

"""
Per-term loss curves over the run (see `LossHistory`).

Log scale on the loss axis: the terms differ by orders of magnitude — a
regularizer at `1e-4` against an L1 term at `1e-2` — and on a linear axis every
curve but the largest is flat against zero, which is precisely the comparison
the plot exists for.

Terms that are identically zero are left out: those are the ones not running,
and an empty line in the legend claims the term is at zero rather than absent.
"""
function loss_plot!(w::RenderWorker)
    history = loss_history(w)
    (history ≡ nothing || isempty(history.steps)) && return

    # ImPlot's defaults (10 px around the plot, 5 px around every label) eat a
    # noticeable share of a docked panel: the curves get that space instead.
    # Pushed before `BeginPlot`, which is where the frame's padding is read.
    ImPlot.PushStyleVar(ImPlot.ImPlotStyleVar_PlotPadding, CImGui.ImVec2(2f0, 2f0))
    ImPlot.PushStyleVar(ImPlot.ImPlotStyleVar_LabelPadding, CImGui.ImVec2(2f0, 2f0))
    ImPlot.PushStyleVar(ImPlot.ImPlotStyleVar_LegendPadding, CImGui.ImVec2(4f0, 4f0))

    # No axis labels: the tick numbers already read as steps & loss, and the
    # two labels cost a text line at the bottom and a rotated one on the left.
    # Both axes refit every frame: the curves grow while training runs, and a
    # plot that keeps the limits of an earlier step would walk off its own data.
    if ImPlot.BeginPlot("##loss-history", "", "",
        CImGui.ImVec2(-1f0, 220f0);
        x_flags=ImPlot.ImPlotAxisFlags_AutoFit,
        y_flags=ImPlot.ImPlotAxisFlags_AutoFit,
    )
        ImPlot.SetupAxisScale(ImPlot.ImAxis_Y1, ImPlot.ImPlotScale_Log10)
        for name in LOSS_FIELDS
            values = getfield(history.terms, name)
            any(!iszero, values) || continue
            ImPlot.PlotLine(String(name), history.steps, values)
        end
        ImPlot.EndPlot()
    end
    # Popped unconditionally: `BeginPlot` returning `false` (a collapsed or
    # clipped panel) still leaves the three vars on the stack.
    ImPlot.PopStyleVar(3)
    return
end

# Two columns: field names left, values right. `id` keeps the tables distinct.
function param_table!(id::String, rows)
    CImGui.BeginTable(id, 2)
    for (name, value) in rows
        CImGui.TableNextRow()
        CImGui.TableNextColumn()
        CImGui.Text(String(name))
        CImGui.TableNextColumn()
        CImGui.Text(format_param(value))
    end
    CImGui.EndTable()
    return
end

"""
`Training Details` as a dockable window: docked beside the controls window on
first use, and detachable — dragged out, it becomes its own OS window.

Only exists while a scene is trainable; viewer-only mode has no
hyperparameters to report.
"""
function training_details_window!(gui, dockspace_id)
    viewer_only(gui) && return

    CImGui.SetNextWindowDockID(dockspace_id, CImGui.ImGuiCond_FirstUseEver)
    CImGui.SetNextWindowSize(
        CImGui.ImVec2(420f0, 600f0), CImGui.ImGuiCond_FirstUseEver)
    if CImGui.Begin("Training Details")
        training_details!(gui)
    end
    CImGui.End()
    return
end

function training_time_line!(w::RenderWorker)
    seconds = w.train_time[]
    steps = w.train_steps[]
    if steps == 0
        CImGui.Text("Train time: —")
        CImGui.SetItemTooltip("Nothing trained yet in this scene.")
        return
    end

    per_step = 1_000.0 * seconds / steps
    CImGui.Text(
        "Train time: $(format_duration(seconds)) " *
        "($(round(per_step; digits=1)) ms/step)")
    CImGui.SetItemTooltip(
        "Wall time spent inside training steps since this scene was loaded, " *
        "over $steps steps.\nExcludes view rendering & time spent paused, and " *
        "a resumed checkpoint starts this back at zero.\nThe first step " *
        "includes GPU kernel compilation, so it weighs on the average early.")
    return
end

"""
What training the scene does at all: run it or not, densify or not, and the two
budgets it stops at. The knobs below are what a run is configured with; these
are what it is doing right now, so they sit above them.
"""
function training_controls!(gui)
    w = gui.worker
    ui_state = gui.ui_state

    CImGui.BeginTable("##training-controls", 2)
    CImGui.TableNextRow()
    CImGui.TableNextColumn()
    # Reflect worker-side stops (e.g. training error)
    # before drawing the checkbox.
    ui_state.train[] = w.train[]
    # Disable `Train` checkbox if no more steps left, until it's raised.
    training_steps_left = w.max_steps[] ≤ 0 || w.step[] < w.max_steps[]
    training_steps_left || disabled_begin()
    if CImGui.Checkbox("Train", ui_state.train)
        GC.gc(false)
        GC.gc(true)
        w.train[] = ui_state.train[]
        notify(w.wakeup)
    end
    training_steps_left || CImGui.SetItemTooltip("Step limit reached: raise `Max Steps` to train further.")
    training_steps_left || disabled_end()

    CImGui.TableNextColumn()
    if CImGui.Checkbox("Densify", ui_state.densify)
        w.densify[] = ui_state.densify[]
    end
    CImGui.SetItemTooltip(
        "Split, clone & prune gaussians on the strategy's schedule below. " *
        "Off freezes the count.")
    CImGui.EndTable()

    CImGui.PushItemWidth(-120)
    if CImGui.InputInt("Max Steps", ui_state.max_steps, 1_000, 5_000)
        ui_state.max_steps[] = max(Int32(0), ui_state.max_steps[])
        w.max_steps[] = Int(ui_state.max_steps[])
        # Raising the limit past the current step lets training resume.
        notify(w.wakeup)
    end
    CImGui.SetItemTooltip("Training stops itself once it has taken this many steps. 0 means no limit.")

    autosave_controls!(gui)

    if ui_state.is_mcmc
        # Benign cross-thread write: `max_cap` is a word-sized Int
        # the worker only reads at densification time.
        strategy = gui.trainer.strategy
        max_cap_ref = Ref{Int32}(strategy.max_cap)
        CImGui.PushItemWidth(-120)
        if CImGui.InputInt("Max Gaussians", max_cap_ref, 100_000, 500_000)
            strategy.max_cap = max(w.n_gaussians[], Int(max_cap_ref[]))
        end
        CImGui.SetItemTooltip(
            "The population MCMC densification grows to; " *
            "never below the count the scene already has.")
    end

    training_time_line!(w)
    return
end

# Where the autosaved checkpoints go, through the same dialog as
# `Save Checkpoint...`. Empty when cancelled.
function pick_autosave_path()
    path = save_file(homedir(); filterlist="safetensors")
    isempty(path) && return ""
    endswith(path, ".safetensors") || (path *= ".safetensors")
    return path
end

"""
Periodic checkpointing: one every N steps & one when the run reaches
`Max Steps`, so a run left alone cannot end with nothing on disk.

Each file is the picked path with its step number appended
(see `autosave_filename`), so they accumulate rather than overwrite.
"""
function autosave_controls!(gui)
    w = gui.worker
    ui_state = gui.ui_state

    # The worker turns this off when a write fails.
    ui_state.autosave[] = w.autosave[]
    if CImGui.Checkbox("Autosave", ui_state.autosave)
        # Nowhere to write yet: ask now, so it cannot be switched on for a run
        # that then saves nothing.
        if ui_state.autosave[] && isempty(ui_state.autosave_path)
            ui_state.autosave_path = pick_autosave_path()
            isempty(ui_state.autosave_path) && (ui_state.autosave[] = false)
            set_autosave_path!(w, ui_state.autosave_path)
        end
        w.autosave[] = ui_state.autosave[]
    end
    CImGui.SetItemTooltip(
        "Save a checkpoint every N steps, and once more when training " *
        "reaches `Max Steps`.")

    ui_state.autosave[] || disabled_begin()

    CImGui.PushItemWidth(-120)
    if CImGui.InputInt("Every N Steps", ui_state.autosave_every, 1_000, 5_000)
        ui_state.autosave_every[] = max(Int32(0), ui_state.autosave_every[])
        w.autosave_every[] = Int(ui_state.autosave_every[])
    end
    CImGui.SetItemTooltip(
        "0 saves only the final step. A save costs a few seconds on a large " *
        "scene & training waits for it.")

    if CImGui.Button("Save To...")
        path = pick_autosave_path()
        if !isempty(path)
            ui_state.autosave_path = path
            set_autosave_path!(w, path)
        end
    end
    CImGui.SameLine()
    # A name a save would actually write, rather than the picked base: the step
    # suffix is what tells the files apart on disk.
    example = isempty(ui_state.autosave_path) ? "" :
        autosave_filename(ui_state.autosave_path, w.step[])
    CImGui.Text(isempty(example) ? "no path set" : basename(example))
    CImGui.SetItemTooltip(isempty(example) ?
        "Pick where the checkpoints go." :
        "Named like `$example`, with the step of each save.")

    ui_state.autosave[] || disabled_end()
    return
end

"""
Contents of the `Training Details` window: the controls that drive training,
then the hyperparameters of the current trainer & its densification strategy.
"""
function training_details!(gui)
    trainer = gui.trainer
    params = trainer.opt_params

    training_controls!(gui)

    CImGui.SeparatorText("Loss")
    loss_plot!(gui.worker)

    CImGui.Separator()
    CImGui.Text("Strategy: $(gui.ui_state.is_mcmc ? "mcmc" : "default")")
    CImGui.Text("Max SH degree: $(gui.ui_state.max_sh_degree)")
    CImGui.Text("Rasterizer mode: $(training_rasterizer_mode(params))")
    CImGui.SetItemTooltip(
        "Channels the training rasterizer blends: the extra ones are what " *
        "depth & normal supervision score against.")

    if CImGui.Button("Save Params...")
        params_file = save_file(homedir(); filterlist="toml") # Empty when cancelled.
        if !isempty(params_file)
            endswith(params_file, ".toml") || (params_file *= ".toml")
            try
                save_opt_params(params_file, params)
            catch err
                @error "Failed to write `OptimizationParams`:" exception=(err, catch_backtrace())
            end
        end
    end
    CImGui.SetItemTooltip(
        "Write these values as a `.toml` file, which `Open Dataset` reads " *
        "back to reproduce this run.")

    for section in OPT_PARAM_SECTIONS
        CImGui.SeparatorText(section.title)

        # An inactive group's values are still reported — they are what the run
        # would use if it were switched on — but greyed out, so a weight that
        # is doing nothing cannot be mistaken for one that is.
        active = section.flag ≡ nothing || getfield(params, section.flag)
        section.flag ≡ nothing || param_table!("##flag-$(section.title)",
            ((section.flag, getfield(params, section.flag)),))

        active || disabled_begin()
        param_table!("##params-$(section.title)",
            ((name, getfield(params, name)) for name in section.fields))
        active || disabled_end()
    end

    # Strategy knobs live on the strategy, not in `OptimizationParams`;
    # skip its per-Gaussian state arrays, which are scene data rather than settings.
    # `max_cap` is left out: `Max Gaussians` above owns it, and the same number
    # twice — once editable, once not — reads as two different settings.
    CImGui.SeparatorText("Densification")
    strategy = trainer.strategy
    param_table!("##strategy-params",
        ((name, getfield(strategy, name))
        for name in fieldnames(typeof(strategy))
        if name ≢ :max_cap && !(getfield(strategy, name) isa AbstractArray)))
    return
end
