#!/usr/bin/env julia
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Julia port of the COLMAP converter script, which itself is based on the
# shell converter script provided in the MipNerF 360 repository.

const USAGE = """
Usage: julia gs-convert.jl -s SOURCE_PATH... [-o OUTPUT_PATH] [options]

Reconstructs a single dataset from one or more sets of input images.
Each SOURCE_PATH is expected to contain an `input` directory with images.
With several sources, their images are staged into OUTPUT_PATH/input as one
dataset (file names are prefixed with the source directory name) and are
reconstructed jointly.

Options:
  -s, --source_path PATH...  Dataset path(s) (required). May be repeated
                             and/or given several values at once.
  -o, --output_path PATH     Where the reconstruction is written. Defaults to
                             the source path; required with several sources.
      --camera MODEL         COLMAP camera model (default: OPENCV).
      --single_camera BOOL   Whether all images share intrinsics (default: 1
                             for a single source, 0 for several sources).
      --matcher NAME         COLMAP matcher: sequential, exhaustive, spatial or
                             vocab_tree (default: sequential for a single
                             source, exhaustive for several sources).
      --num_threads N        Limit feature extraction, matching and mapping to
                             N threads (default: let COLMAP use all cores).
      --colmap_executable P  Path to the `colmap` executable (default: colmap).
      --magick_executable P  Path to the ImageMagick executable
                             (default: invoke `mogrify` directly).
      --copy_images          Copy staged images instead of symlinking them.
      --skip_matching        Skip feature extraction, matching and mapping.
      --resize               Also write 1/2, 1/4 and 1/8 scale images.
      --no_gpu               Accepted for compatibility, GPU is never used.
  -h, --help                 Show this message.
"""

const MATCHERS = ("sequential", "exhaustive", "spatial", "vocab_tree")

struct Args
    source_paths::Vector{String}
    output_path::String
    camera::String
    single_camera::String
    matcher::String
    num_threads::Int
    colmap_executable::String
    magick_executable::String
    copy_images::Bool
    skip_matching::Bool
    resize::Bool
    no_gpu::Bool
end

function parse_args(argv::Vector{String})
    source_paths = String[]
    output_path = ""
    camera = "OPENCV"
    single_camera = ""
    matcher = ""
    num_threads = -1  # COLMAP's own default: use every core.
    colmap_executable = ""
    magick_executable = ""
    copy_images = false
    skip_matching = false
    resize = false
    no_gpu = false

    # Fetches the single value of an option, erroring out if it is missing.
    function value(i, flag)
        i + 1 > length(argv) && error("$flag requires a value.")
        argv[i + 1]
    end

    # Fetches all values of an option, up to the next flag.
    function multivalue(i, flag)
        last = i
        while last + 1 ≤ length(argv) && !startswith(argv[last + 1], "-")
            last += 1
        end
        last == i && error("$flag requires at least one value.")
        argv[(i + 1):last]
    end

    i = 1
    while i ≤ length(argv)
        arg = argv[i]
        if arg == "-s" || arg == "--source_path"
            paths = multivalue(i, arg)
            append!(source_paths, paths); i += length(paths) + 1
        elseif arg == "-o" || arg == "--output_path"
            output_path = value(i, arg); i += 2
        elseif arg == "--camera"
            camera = value(i, arg); i += 2
        elseif arg == "--single_camera"
            single_camera = value(i, arg); i += 2
        elseif arg == "--matcher"
            matcher = value(i, arg); i += 2
        elseif arg == "--num_threads"
            raw = value(i, arg)
            num_threads = tryparse(Int, raw)
            num_threads ≡ nothing && error("--num_threads expects an integer, got: `$raw`")
            num_threads > 0 || error("--num_threads must be positive, got: $num_threads")
            i += 2
        elseif arg == "--colmap_executable"
            colmap_executable = value(i, arg); i += 2
        elseif arg == "--magick_executable"
            magick_executable = value(i, arg); i += 2
        elseif arg == "--copy_images"
            copy_images = true; i += 1
        elseif arg == "--skip_matching"
            skip_matching = true; i += 1
        elseif arg == "--resize"
            resize = true; i += 1
        elseif arg == "--no_gpu"
            no_gpu = true; i += 1
        elseif arg == "-h" || arg == "--help"
            print(USAGE)
            exit(0)
        else
            error("Unrecognized argument: $arg\n\n$USAGE")
        end
    end

    isempty(source_paths) && error("--source_path/-s is required.\n\n$USAGE")
    multiple = length(source_paths) > 1

    for path in source_paths
        isdir(joinpath(path, "input")) ||
            error("Source path has no `input` directory: $path")
    end

    if isempty(output_path)
        multiple && error(
            "--output_path/-o is required when several sources are given.")
        output_path = source_paths[1]
    elseif multiple
        # Staging into a source would mix prefixed copies into its own input.
        for path in source_paths
            realpath_eq(path, output_path) && error(
                "--output_path must differ from every source path, got: $path")
        end
    end

    isempty(single_camera) && (single_camera = multiple ? "0" : "1")
    isempty(matcher) && (matcher = multiple ? "exhaustive" : "sequential")
    matcher in MATCHERS || error(
        "Unknown matcher `$matcher`, expected one of: $(join(MATCHERS, ", ")).")

    Args(
        source_paths, output_path, camera, single_camera, matcher, num_threads,
        colmap_executable, magick_executable, copy_images, skip_matching,
        resize, no_gpu)
end

# Compares two paths by identity, tolerating non-existent ones.
function realpath_eq(a::AbstractString, b::AbstractString)
    resolve(p) = ispath(p) ? realpath(p) : abspath(p)
    resolve(a) == resolve(b)
end

# Derives a unique file name prefix for every source dataset.
function dataset_prefixes(source_paths::Vector{String})
    names = map(source_paths) do path
        name = basename(rstrip(path, ['/', '\\']))
        isempty(name) ? "dataset" : name
    end
    duplicates = filter(n -> count(==(n), names) > 1, names)
    map(enumerate(names)) do (i, name)
        name in duplicates ? "$(name)_$i" : name
    end
end

# Symlinks (or copies) `src` to `dst`, replacing whatever is at `dst`.
function link_or_copy(src::AbstractString, dst::AbstractString, copy::Bool)
    (islink(dst) || ispath(dst)) && rm(dst)
    if copy
        cp(src, dst)
    else
        try
            symlink(abspath(src), dst)
        catch
            # Filesystems without symlink support (or permission for them).
            cp(src, dst)
        end
    end
    return
end

# Gathers the images of every source into a single `input` directory,
# prefixing file names with the dataset they came from to avoid collisions.
function stage_inputs(args::Args)
    input_path = joinpath(args.output_path, "input")
    mkpath(input_path)

    total = 0
    for (prefix, source) in zip(dataset_prefixes(args.source_paths), args.source_paths)
        source_input = joinpath(source, "input")
        staged = 0
        for file in readdir(source_input)
            source_file = joinpath(source_input, file)
            isfile(source_file) || continue
            link_or_copy(
                source_file, joinpath(input_path, "$(prefix)_$file"),
                args.copy_images)
            staged += 1
        end
        staged == 0 && @warn "No images found in $source_input."
        total += staged
        println("  $source → $staged image(s) as `$(prefix)_*`.")
    end
    println("Staged $total image(s) from " *
        "$(length(args.source_paths)) datasets into $input_path.")

    return input_path
end

# Runs `cmd`, exiting the process with its exit code if it fails.
function run_or_exit(cmd::Cmd, what::AbstractString)
    process = run(ignorestatus(cmd); wait=true)
    code = process.exitcode
    if code != 0
        @error "$what failed with code $code. Exiting."
        exit(code)
    end
    return
end

function main(argv::Vector{String})
    args = parse_args(argv)
    colmap_command = isempty(args.colmap_executable) ?
        "colmap" : args.colmap_executable
    magick_command = isempty(args.magick_executable) ?
        `mogrify` : `$(args.magick_executable) mogrify`
    use_gpu = 0

    output_path = args.output_path
    database_path = joinpath(output_path, "distorted", "database.db")
    # A lone source is used as-is, several are merged into `output/input`.
    image_path = length(args.source_paths) == 1 ?
        joinpath(args.source_paths[1], "input") : stage_inputs(args)

    if !args.skip_matching
        mkpath(joinpath(output_path, "distorted", "sparse"))

        # Every stage names its thread count differently. Left out entirely
        # when unset, so that COLMAP keeps picking the count itself.
        threads(option) = args.num_threads > 0 ?
            [option, string(args.num_threads)] : String[]

        # Feature extraction.
        run_or_exit(`$colmap_command feature_extractor
            --database_path $database_path
            --image_path $image_path
            --ImageReader.single_camera $(args.single_camera)
            --ImageReader.camera_model $(args.camera)
            --SiftExtraction.use_gpu $use_gpu
            $(threads("--SiftExtraction.num_threads"))`, "Feature extraction")

        # Feature matching.
        run_or_exit(`$colmap_command $(args.matcher)_matcher
            --database_path $database_path
            --SiftMatching.use_gpu $use_gpu
            $(threads("--SiftMatching.num_threads"))`, "Feature matching")

        # Bundle adjustment.
        # The default Mapper tolerance is unnecessarily large,
        # decreasing it speeds up bundle adjustment steps.
        run_or_exit(`$colmap_command mapper
            --database_path $database_path
            --image_path $image_path
            --output_path $(joinpath(output_path, "distorted", "sparse"))
            --Mapper.ba_global_function_tolerance=0.000001
            $(threads("--Mapper.num_threads"))`, "Mapper")
    end

    # Image undistortion.
    # We need to undistort our images into ideal pinhole intrinsics.
    run_or_exit(`$colmap_command image_undistorter
        --image_path $image_path
        --input_path $(joinpath(output_path, "distorted", "sparse", "0"))
        --output_path $output_path
        --output_type COLMAP`, "Image undistortion")

    sparse_path = joinpath(output_path, "sparse")
    files = readdir(sparse_path)
    mkpath(joinpath(sparse_path, "0"))
    # Move each file from the source directory to the destination directory.
    for file in files
        file == "0" && continue
        mv(joinpath(sparse_path, file), joinpath(sparse_path, "0", file);
            force=true)
    end

    if args.resize
        println("Copying and resizing...")

        images_path = joinpath(output_path, "images")
        scales = ((2, "50%"), (4, "25%"), (8, "12.5%"))
        for (scale, _) in scales
            mkpath(joinpath(output_path, "images_$scale"))
        end

        for file in readdir(images_path)
            source_file = joinpath(images_path, file)
            for (scale, percentage) in scales
                destination_file = joinpath(
                    output_path, "images_$scale", file)
                cp(source_file, destination_file; force=true)
                run_or_exit(
                    `$magick_command -resize $percentage $destination_file`,
                    "$percentage resize")
            end
        end
    end

    println("Done.")
    return
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
