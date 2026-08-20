#!/usr/bin/env julia
#
# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Converts a RealityCapture / RealityScan export into the COLMAP layout the
# training path reads. The counterpart of `gs-convert.jl`, which does the same
# job for raw images by running COLMAP itself.

include(joinpath(@__DIR__, "realitycapture.jl"))

const USAGE = """
Usage: julia --project scripts/rc-convert.jl -s SOURCE_PATH [-o OUTPUT_PATH] [options]

Converts a RealityCapture / RealityScan export into a COLMAP dataset.
SOURCE_PATH is expected to contain:

  input/             the images, as exported
  camera-params.csv  the camera parameters, as exported
  points.ply         the point cloud, exported with XYZ & RGB

The images are undistorted into a single shared pinhole & written to
OUTPUT_PATH/images, alongside an OUTPUT_PATH/sparse/0 COLMAP model.

Options:
  -s, --source_path PATH  RealityCapture export path (required).
  -o, --output_path PATH  Where the conversion is written. Defaults to the
                          source path.
      --resize            Also write 1/2, 1/4 and 1/8 scale images.
      --max_points N      Subsample the initial point cloud to N points.
      --focal_tolerance F Drop views whose focal deviates from the dataset
                          median by more than this fraction (default: 0.02).
                          Zoomed-in frames cannot be warped into the shared
                          pinhole without leaving a black border.
  -h, --help              Show this message.
"""

function main(argv::Vector{String})
    source_path = ""
    output_path = ""
    resize = false
    max_points = 0
    focal_tolerance = -1.0

    function value(i, flag)
        i + 1 > length(argv) && error("$flag requires a value.")
        argv[i + 1]
    end
    function number(::Type{T}, i, flag) where {T}
        raw = value(i, flag)
        parsed = tryparse(T, raw)
        parsed ≡ nothing && error("$flag expects a $T, got: `$raw`.")
        parsed
    end

    i = 1
    while i ≤ length(argv)
        arg = argv[i]
        if arg == "-s" || arg == "--source_path"
            source_path = value(i, arg); i += 2
        elseif arg == "-o" || arg == "--output_path"
            output_path = value(i, arg); i += 2
        elseif arg == "--resize"
            resize = true; i += 1
        elseif arg == "--max_points"
            max_points = number(Int, i, arg); i += 2
        elseif arg == "--focal_tolerance"
            focal_tolerance = number(Float64, i, arg); i += 2
        elseif arg == "-h" || arg == "--help"
            print(USAGE); exit(0)
        else
            error("Unrecognized argument: $arg\n\n$USAGE")
        end
    end
    isempty(source_path) && error("--source_path/-s is required.\n\n$USAGE")
    isempty(output_path) && (output_path = source_path)

    kwargs = (; output_dir=output_path, resize, max_points)
    focal_tolerance > 0 && (kwargs = (; kwargs..., focal_tolerance))
    rc_convert(source_path; kwargs...)

    println("Done.")
    return
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
