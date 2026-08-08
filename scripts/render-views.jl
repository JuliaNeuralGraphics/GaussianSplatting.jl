#!/usr/bin/env julia
#
# Renders a trained model from every camera pose of a COLMAP dataset, writing
# an RGB image, a depth map and a normal map per view.
#
# All three come out of a single `:rgbdn` rasterization, whose channels are
# `rgb | depth | alpha | normal` (see `n_color_features`). Depth & normal are
# alpha-weighted sums, so both are divided by the alpha map before they mean
# anything per-pixel: `D / α` is the expected depth along the ray & `N / α` the
# expected surface normal. Pixels the model does not cover (`α ≈ 0`) carry no
# geometry and are written as black / neutral rather than as a ratio of two
# near-zeros.
#
# The dataset supplies the poses & the intrinsics only - the images it loads
# are not used here, but `ColmapDataset` is what reads `sparse/0`, so they are
# paid for anyway.

const USAGE = """
Usage: julia --project scripts/render-views.jl -d DATASET -m MODEL [-o OUTPUT] [options]

Renders MODEL from every camera pose of DATASET into OUTPUT:

    OUTPUT/rgb/<view>.png       8-bit color render
    OUTPUT/depth/<view>.png     16-bit grayscale expected depth
    OUTPUT/normal/<view>.png    8-bit normal map, `0.5 · (n + 1)` encoded
    OUTPUT/depth-scales.csv     scene-unit depth each map's white level means

Depth is normalized per view by default, which is what makes a single map
readable; pass --depth-max to put every view on one scale instead (needed to
compare views against each other or to assemble them into a video).

Julia must be started in an environment that has both GaussianSplatting and a
GPU backend package (CUDA, AMDGPU or Metal) installed.

Options:
  -d, --dataset PATH      COLMAP dataset directory (required).
  -m, --model PATH        Trained model: a `.ply` or a `.safetensors`
                          checkpoint (required).
  -o, --output PATH       Where renders are written
                          (default: DATASET/renders).
  -s, --scale N           Dataset image scale, i.e. render at the resolution of
                          `images_N` (default: 1).
      --split NAME        Poses to render: all (default), train or test.
      --holdout N         Every N-th view (by file name) is the test split;
                          only used by --split train/test (default: 8).
      --backend NAME      cuda, amdgpu or metal
                          (default: the first one that is installed & working).
      --depth-max VALUE   Scene-unit depth mapped to white, shared by every
                          view. Default: normalize each view by its own
                          --depth-percentile.
      --depth-percentile P
                          Percentile of a view's covered depths mapped to
                          white when --depth-max is not given (default: 99).
      --world-normals     Write world-space normals instead of the rendered
                          camera-space ones.
      --skip-existing     Leave views whose three files are all present alone.
  -h, --help              Show this message.
"""

import GaussianSplatting as GSP
using GaussianSplatting.FileIO
using GaussianSplatting.ImageCore
using GaussianSplatting.Statistics
import GaussianSplatting.ProgressMeter

# Coverage below this is background: the alpha-normalized depth & normal there
# divide two near-zeros, so they are masked out instead.
const MIN_ALPHA = 1f-3

const SPLITS = ("all", "train", "test")
const BACKENDS = ("cuda", "amdgpu", "metal")

struct Args
    dataset::String
    model::String
    output::String
    scale::Int
    split::String
    holdout::Int
    backend::String
    depth_max::Float32
    depth_percentile::Float64
    world_normals::Bool
    skip_existing::Bool
end

function parse_args(argv::Vector{String})
    dataset = ""
    model = ""
    output = ""
    scale = 1
    split = "all"
    holdout = 8
    backend = ""
    depth_max = 0f0 # 0 = per-view normalization.
    depth_percentile = 99.0
    world_normals = false
    skip_existing = false

    # Fetches the single value of an option, erroring out if it is missing.
    function value(i, flag)
        i + 1 > length(argv) && error("$flag requires a value.")
        argv[i + 1]
    end

    function number(i, flag, ::Type{T}) where T <: Number
        raw = value(i, flag)
        parsed = tryparse(T, raw)
        parsed ≡ nothing && error("$flag expects a $T, got: `$raw`")
        parsed
    end

    i = 1
    while i ≤ length(argv)
        arg = argv[i]
        if arg == "-d" || arg == "--dataset"
            dataset = value(i, arg); i += 2
        elseif arg == "-m" || arg == "--model"
            model = value(i, arg); i += 2
        elseif arg == "-o" || arg == "--output"
            output = value(i, arg); i += 2
        elseif arg == "-s" || arg == "--scale"
            scale = number(i, arg, Int)
            scale > 0 || error("--scale must be positive, got: $scale")
            i += 2
        elseif arg == "--split"
            split = lowercase(value(i, arg)); i += 2
        elseif arg == "--holdout"
            holdout = number(i, arg, Int)
            holdout > 1 || error("--holdout must be greater than 1, got: $holdout")
            i += 2
        elseif arg == "--backend"
            backend = lowercase(value(i, arg)); i += 2
        elseif arg == "--depth-max"
            depth_max = number(i, arg, Float32)
            depth_max > 0f0 || error("--depth-max must be positive, got: $depth_max")
            i += 2
        elseif arg == "--depth-percentile"
            depth_percentile = number(i, arg, Float64)
            0 < depth_percentile ≤ 100 || error(
                "--depth-percentile must be in `(0, 100]`, got: $depth_percentile")
            i += 2
        elseif arg == "--world-normals"
            world_normals = true; i += 1
        elseif arg == "--skip-existing"
            skip_existing = true; i += 1
        elseif arg == "-h" || arg == "--help"
            print(USAGE)
            exit(0)
        else
            error("Unrecognized argument: $arg\n\n$USAGE")
        end
    end

    isempty(dataset) && error("--dataset/-d is required.\n\n$USAGE")
    isempty(model) && error("--model/-m is required.\n\n$USAGE")
    isdir(dataset) || error("Dataset directory does not exist: `$dataset`.")
    isfile(model) || error("Model file does not exist: `$model`.")
    split in SPLITS || error(
        "Unknown split `$split`, expected one of: $(join(SPLITS, ", ")).")
    isempty(backend) || backend in BACKENDS || error(
        "Unknown backend `$backend`, expected one of: $(join(BACKENDS, ", ")).")
    isempty(output) && (output = joinpath(dataset, "renders"))

    Args(
        dataset, model, output, scale, split, holdout, backend,
        depth_max, depth_percentile, world_normals, skip_existing)
end

"""
Load the GPU package `name` & return its `KernelAbstractions` backend.

The package is loaded into `Main` at run time rather than `using`-ed at the
top of the script, so that only the one backend a machine actually has is
required: loading it there is also what makes GaussianSplatting's extension
for it kick in.

Everything it brings in - `functional`, the backend type, the extension's
methods - is younger than the frame that loads it, hence `invokelatest` here &
around [`render`](@ref), which runs on top of those extension methods.
"""
function backend_by_name(name::String)
    package, backend = if name == "cuda"
        @eval Main import CUDA
        :CUDA, :CUDABackend
    elseif name == "amdgpu"
        @eval Main import AMDGPU
        :AMDGPU, :ROCBackend
    elseif name == "metal"
        @eval Main import Metal
        :Metal, :MetalBackend
    else
        error("Unknown backend: `$name`.")
    end

    # Everything the import defined - the module binding included - belongs to
    # a world younger than this frame's, so it may only be reached from a call
    # that runs in the latest one.
    return invokelatest() do
        mod = getglobal(Main, package)
        getglobal(mod, :functional)() ||
            error("`$name` is installed but not functional.")
        return getglobal(mod, backend)()
    end
end

# Package a backend name refers to, as `Base.find_package` spells it.
backend_package(name::String) =
    name == "cuda" ? "CUDA" : name == "amdgpu" ? "AMDGPU" : "Metal"

"""
Backend to render on: the requested one, or the first installed & working one.
"""
function select_backend(name::String)
    isempty(name) || return backend_by_name(name)

    for candidate in BACKENDS
        Base.find_package(backend_package(candidate)) ≡ nothing && continue
        try
            return backend_by_name(candidate)
        catch e
            @warn "Skipping `$candidate` backend." exception=e
        end
    end
    error(
        "No usable GPU backend found. Install one of " *
        "$(join(BACKENDS, ", ")) in the active environment " *
        "(`] add CUDA`) or select it with --backend.")
end

"""
Load the trained Gaussians from a `.ply` or a `.safetensors` checkpoint.

A checkpoint may carry a sky dome, which is part of the scene's appearance &
is folded into the model here, exactly as `export_ply` folds it in.
"""
function load_model(kab, path::String)
    if endswith(lowercase(path), ".ply")
        return GSP.import_ply(path, kab).gaussians
    end
    return GSP.load_checkpoint_model(kab, path).gaussians
end

"""
Cameras of `dataset` to render & the file names they were reconstructed from.

`ColmapDataset` applies the `holdout` split itself, so `all` simply asks it
not to hold anything out. Views come back in file-name order in every case.
"""
function select_views(dataset, split::String)
    split == "test" && return dataset.test_cameras, dataset.test_image_filenames
    return dataset.train_cameras, dataset.train_image_filenames
end

"""
Encode the alpha-normalized expected depth `D / α` as a 16-bit grayscale image
& report the scene-unit depth its white level stands for.

`depth_max` fixes that scale across views; `0` derives it per view from the
`percentile`-th covered depth, so that a handful of far outliers (a stray
floater behind the scene) do not push the whole map into the first few
gray levels. Uncovered pixels are black.
"""
function depth_image(
    depth::Matrix{Float32}, alpha::Matrix{Float32};
    depth_max::Float32, percentile::Float64,
)
    covered = alpha .≥ MIN_ALPHA
    expected = depth ./ max.(alpha, MIN_ALPHA)
    expected[.!covered] .= 0f0

    scale::Float32 = if depth_max > 0f0
        depth_max
    else
        values = expected[covered]
        isempty(values) ? 1f0 : max(Float32(quantile(values, percentile / 100)), 1f-6)
    end

    normalized = clamp01!(expected .* (1f0 / scale))
    return colorview(Gray, N0f16.(permutedims(normalized, (2, 1)))), scale
end

"""
Encode the alpha-normalized expected normal as an 8-bit image, mapping the
unit vector `n` to `0.5 · (n + 1)`.

Rendered normals are camera-space & already oriented towards the camera (see
`gaussian_normal`); `R_c2w` rotates them into world space instead when given.
Uncovered pixels & degenerate sums encode the zero vector, i.e. neutral gray.
"""
function normal_image(
    normals::Array{Float32, 3}, alpha::Matrix{Float32}; R_c2w = nothing,
)
    width, height = size(alpha)
    n = reshape(normals, 3, :) ./ reshape(max.(alpha, MIN_ALPHA), 1, :)

    # Blending shortens the summed normal, so it has to be re-normalized
    # rather than merely scaled by `1 / α`.
    lengths = sqrt.(sum(abs2, n; dims=1))
    degenerate = vec((lengths .< 1f-6) .| reshape(alpha .< MIN_ALPHA, 1, :))
    n ./= max.(lengths, 1f-6)
    n[:, degenerate] .= 0f0

    R_c2w ≡ nothing || (n = Matrix{Float32}(R_c2w) * n)

    encoded = reshape(n .* 0.5f0 .+ 0.5f0, 3, width, height)
    return colorview(RGB, permutedims(encoded, (1, 3, 2)))
end

"""
Depth white levels a previous run recorded, by view name.

Only `--skip-existing` needs them: the table is rewritten from scratch, and a
resumed run has to put back the rows of the views it did not re-render.
A missing or malformed file simply contributes nothing.
"""
function read_depth_scales(path::String)
    scales = Dict{String, Float32}()
    isfile(path) || return scales

    for line in Iterators.drop(eachline(path), 1) # Header.
        # Split at the *last* separator: view names come from image file names,
        # which nothing stops from containing a comma themselves.
        i = findlast(==(','), line)
        i ≡ nothing && continue
        name = line[1:prevind(line, i)]
        scale = tryparse(Float32, line[nextind(line, i):end])
        (isempty(name) || scale ≡ nothing) && continue
        scales[name] = scale
    end
    return scales
end

"""
Render every selected view of the dataset onto disk.

Called through `invokelatest` (see [`backend_by_name`](@ref)): it runs on
methods the GPU backend & its GaussianSplatting extension defined after this
script started.
"""
function render(args::Args, kab)
    gaussians = load_model(kab, args.model)
    @info "Loaded $(length(gaussians)) Gaussians from `$(args.model)` " *
        "(SH degree $(gaussians.sh_degree))."

    # `all` renders every pose, so nothing may be held out.
    holdout = args.split == "all" ? 0 : args.holdout
    dataset = GSP.ColmapDataset(args.dataset; scale=args.scale, holdout)
    cameras, filenames = select_views(dataset, args.split)
    isempty(cameras) && error(
        "No cameras to render for `--split $(args.split)`.")

    # All dataset cameras share intrinsics, so one rasterizer serves them all.
    (; width, height) = GSP.resolution(cameras[1])
    # `:rgbdn` is what renders the depth, alpha & normal channels at all.
    rasterizer = GSP.GaussianRasterizer(kab, cameras[1]; mode=:rgbdn)

    rgb_dir = joinpath(args.output, "rgb")
    depth_dir = joinpath(args.output, "depth")
    normal_dir = joinpath(args.output, "normal")
    foreach(mkpath, (rgb_dir, depth_dir, normal_dir))

    @info "Rendering $(length(cameras)) `$(args.split)` views at " *
        "($width x $height) into `$(args.output)`."

    scales_file = joinpath(args.output, "depth-scales.csv")
    previous_scales = args.skip_existing ?
        read_depth_scales(scales_file) : Dict{String, Float32}()

    depth_scales = Tuple{String, Float32}[]
    skipped = 0
    meter = ProgressMeter.Progress(length(cameras); desc="Rendering: ", showspeed=true)
    for (camera, filename) in zip(cameras, filenames)
        name = first(splitext(basename(filename)))
        rgb_file = joinpath(rgb_dir, "$name.png")
        depth_file = joinpath(depth_dir, "$name.png")
        normal_file = joinpath(normal_dir, "$name.png")

        if args.skip_existing && all(isfile, (rgb_file, depth_file, normal_file))
            skipped += 1
            # Carry the scale of the map that is already on disk over into the
            # rewritten table, so a resumed run does not drop the rows of the
            # views it skipped.
            haskey(previous_scales, name) &&
                push!(depth_scales, (name, previous_scales[name]))
            ProgressMeter.next!(meter)
            continue
        end

        features = rasterizer(
            gaussians.points, gaussians.opacities, gaussians.scales,
            gaussians.rotations, gaussians.features_dc, gaussians.features_rest;
            camera, sh_degree=gaussians.sh_degree)
        # One transfer for all three maps: the rasterizer reuses `features`
        # for the next view, so nothing may be read from it lazily.
        host = Array(features)
        alpha = host[5, :, :]

        save(rgb_file, GSP.to_image(host[1:3, :, :]))

        depth, scale = depth_image(host[4, :, :], alpha;
            depth_max=args.depth_max, percentile=args.depth_percentile)
        save(depth_file, depth)
        push!(depth_scales, (name, scale))

        save(normal_file, normal_image(host[6:8, :, :], alpha;
            R_c2w=args.world_normals ? transpose(camera.R) : nothing))

        ProgressMeter.next!(meter)
    end
    ProgressMeter.finish!(meter)

    # The white level of a depth map is meaningless without this: with per-view
    # normalization it differs from view to view.
    open(scales_file, "w") do io
        println(io, "view,depth_scale")
        for (name, scale) in depth_scales
            println(io, "$name,$scale")
        end
    end

    skipped > 0 && @info "Skipped $skipped already rendered view(s)."
    @info "Rendered $(length(depth_scales)) view(s) into `$(args.output)`. " *
        "Depth white levels: `$scales_file`."
    return
end

function main(argv::Vector{String})
    args = parse_args(argv)
    kab = select_backend(args.backend)
    @info "Using `$kab` GPU backend."
    invokelatest(render, args, kab)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
