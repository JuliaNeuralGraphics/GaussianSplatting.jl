#!/usr/bin/env julia
#
# Extracts a set of sharp frames from a video, ready to be reconstructed with `gs-convert.jl`.
#
# FFmpeg's `blurdetect` filter scores every frame (higher score = blurrier),
# so the video is decoded twice: once to collect the scores, once to write out only
# the frames that are worth keeping.
#
# Rather than thresholding the score, frames are grouped into fixed-length
# time windows and the sharpest frame of each window is kept. This keeps the
# camera path evenly covered - an absolute threshold would take hundreds of
# frames from a slow, well-lit part of the sweep and none from a fast one.

const USAGE = """
Usage: julia extract-frames.jl -i VIDEO [-o OUTPUT_PATH] [options]

Extracts the sharpest frame of every time window of the video into
OUTPUT_PATH/input, which `gs-convert.jl -s OUTPUT_PATH` then reconstructs.

Options:
  -i, --video PATH           Input video (required).
  -o, --output_path PATH     Dataset directory to create. Frames are written
                             to its `input` subdirectory. Defaults to the
                             video path without its extension.
  -w, --window SECONDS       Keep the sharpest frame of every window of this
                             length (default: 0.5).
  -n, --num_frames N         Target frame count. Picks the window length from
                             the video duration instead of --window.
      --blur_threshold F     Also drop kept frames whose blur score exceeds F,
                             leaving those windows empty. Off by default; run
                             once without it and look at the reported scores.
      --start SECONDS        Skip everything before this timestamp.
      --end SECONDS          Skip everything after this timestamp.
      --format EXT           `png` (default) or `jpg`.
      --quality Q            JPEG quality, 2 (best) to 31 (default: 2).
      --max_width PIXELS     Downscale frames wider than this, keeping aspect.
      --block_pct PCT        Score frames on their sharpest PCT percent of
                             blocks, so that a blurry background does not hide
                             a well focused subject (default: 90).
      --ffmpeg_executable P  Path to the `ffmpeg` executable (default: ffmpeg).
      --dry_run              Report what would be kept without writing frames.
  -h, --help                 Show this message.
"""

const FORMATS = ("png", "jpg")

struct Args
    video::String
    output_path::String
    window::Float64
    num_frames::Int
    blur_threshold::Float64
    start_time::Float64
    end_time::Float64
    format::String
    quality::Int
    max_width::Int
    block_pct::Int
    ffmpeg_executable::String
    dry_run::Bool
end

function parse_args(argv::Vector{String})
    video = ""
    output_path = ""
    window = 0.5
    num_frames = -1
    blur_threshold = Inf
    start_time = -1.0
    end_time = -1.0
    format = "png"
    quality = 2
    max_width = -1
    block_pct = 90
    ffmpeg_executable = ""
    dry_run = false

    # Fetches the single value of an option, erroring out if it is missing.
    function value(i, flag)
        i + 1 > length(argv) && error("$flag requires a value.")
        argv[i + 1]
    end

    function number(i, flag, T)
        raw = value(i, flag)
        parsed = tryparse(T, raw)
        parsed ≡ nothing && error("$flag expects a number, got: `$raw`")
        parsed
    end

    i = 1
    while i ≤ length(argv)
        arg = argv[i]
        if arg == "-i" || arg == "--video"
            video = value(i, arg); i += 2
        elseif arg == "-o" || arg == "--output_path"
            output_path = value(i, arg); i += 2
        elseif arg == "-w" || arg == "--window"
            window = number(i, arg, Float64); i += 2
        elseif arg == "-n" || arg == "--num_frames"
            num_frames = number(i, arg, Int); i += 2
        elseif arg == "--blur_threshold"
            blur_threshold = number(i, arg, Float64); i += 2
        elseif arg == "--start"
            start_time = number(i, arg, Float64); i += 2
        elseif arg == "--end"
            end_time = number(i, arg, Float64); i += 2
        elseif arg == "--format"
            format = lowercase(value(i, arg)); i += 2
        elseif arg == "--quality"
            quality = number(i, arg, Int); i += 2
        elseif arg == "--max_width"
            max_width = number(i, arg, Int); i += 2
        elseif arg == "--block_pct"
            block_pct = number(i, arg, Int); i += 2
        elseif arg == "--ffmpeg_executable"
            ffmpeg_executable = value(i, arg); i += 2
        elseif arg == "--dry_run"
            dry_run = true; i += 1
        elseif arg == "-h" || arg == "--help"
            print(USAGE)
            exit(0)
        else
            error("Unrecognized argument: $arg\n\n$USAGE")
        end
    end

    isempty(video) && error("--video/-i is required.\n\n$USAGE")
    isfile(video) || error("Video does not exist: $video")

    format == "jpeg" && (format = "jpg")
    format in FORMATS || error(
        "Unknown format `$format`, expected one of: $(join(FORMATS, ", ")).")

    num_frames == 0 && error("--num_frames must be positive.")
    num_frames < 0 && window ≤ 0 && error("--window must be positive.")
    1 ≤ quality ≤ 31 || error("--quality must be in 1..31, got: $quality")
    0 < block_pct ≤ 100 || error("--block_pct must be in 1..100, got: $block_pct")
    max_width == 0 && error("--max_width must be positive.")
    start_time ≥ 0 && end_time ≥ 0 && end_time ≤ start_time &&
        error("--end must be greater than --start.")

    if isempty(output_path)
        # `video.mp4` becomes the `video` dataset next to it.
        base, _ = splitext(video)
        output_path = base
        realpath_eq(output_path, video) && error(
            "Cannot derive an output path from $video, pass --output_path.")
    end

    Args(
        video, output_path, window, num_frames, blur_threshold, start_time,
        end_time, format, quality, max_width, block_pct, ffmpeg_executable,
        dry_run)
end

# Compares two paths by identity, tolerating non-existent ones.
function realpath_eq(a::AbstractString, b::AbstractString)
    resolve(p) = ispath(p) ? realpath(p) : abspath(p)
    resolve(a) == resolve(b)
end

# Escapes a path so that it survives as a filter option value.
escape_filter_path(path::AbstractString) =
    replace(path, "\\" => "\\\\", ":" => "\\:", "'" => "\\'")

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

function check_ffmpeg(ffmpeg_command::AbstractString)
    filters = try
        read(`$ffmpeg_command -hide_banner -filters`, String)
    catch e
        e isa Base.IOError || e isa Base.ProcessFailedException || rethrow()
        error("Could not run `$ffmpeg_command`, is FFmpeg installed?")
    end
    occursin("blurdetect", filters) || error(
        "This FFmpeg build has no `blurdetect` filter, it needs FFmpeg ≥ 5.1.")
    return
end

# Trimming has to be identical in both passes, otherwise the frame indices
# collected by the first one address different frames in the second.
function trim_options(args::Args)
    options = String[]
    args.start_time ≥ 0 && append!(options, ["-ss", string(args.start_time)])
    args.end_time ≥ 0 && append!(options, ["-to", string(args.end_time)])
    options
end

struct Frame
    index::Int
    time::Float64
    blur::Float64
end

# Scores every frame of the video, without encoding anything.
function score_frames(args::Args, ffmpeg_command::AbstractString)
    metadata_path = joinpath(mktempdir(), "blur.txt")
    graph = "blurdetect=block_pct=$(args.block_pct)," *
        "metadata=print:file=$(escape_filter_path(metadata_path))"

    println("Scoring frames of $(args.video)...")
    run_or_exit(`$ffmpeg_command -hide_banner -loglevel warning -nostdin
        $(trim_options(args)) -i $(args.video)
        -map 0:v:0 -an -sn -dn
        -vf $graph -f null -`, "Blur detection")

    isfile(metadata_path) || error("FFmpeg produced no blur scores.")
    parse_scores(metadata_path)
end

# Reads back what `metadata=print` wrote, which alternates between a frame
# header and the metadata keys attached to that frame:
#
#   frame:0    pts:0       pts_time:0
#   lavfi.blur=3.145062
function parse_scores(metadata_path::AbstractString)
    frames = Frame[]
    index, time = -1, NaN
    for line in eachline(metadata_path)
        header = match(r"^frame:(\d+)\s+pts:\S+\s+pts_time:(\S+)", line)
        if header ≢ nothing
            index = parse(Int, header[1])
            # Streams without timestamps report `N/A`, frame indices still
            # order them correctly.
            time = something(tryparse(Float64, header[2]), NaN)
            continue
        end

        blur = match(r"^lavfi\.blur=(\S+)", strip(line))
        blur ≡ nothing && continue
        index < 0 && continue
        push!(frames, Frame(index, time, parse(Float64, blur[1])))
        index = -1
    end

    isempty(frames) && error(
        "No blur scores were parsed, the video may have no video stream.")
    frames
end

# Keeps the sharpest frame of every window of `window` seconds.
function select_frames(frames::Vector{Frame}, window::Float64, threshold::Float64)
    origin = frames[1].time
    # Timestamp-less streams fall back to grouping by frame index.
    bucket(frame) = isnan(frame.time) ?
        fld(frame.index - frames[1].index, max(1, round(Int, window))) :
        fld(frame.time - origin, window)

    best = Dict{Int, Frame}()
    for frame in frames
        key = Int(bucket(frame))
        current = get(best, key, nothing)
        (current ≡ nothing || frame.blur < current.blur) && (best[key] = frame)
    end

    selected = sort!(collect(values(best)); by=frame -> frame.index)
    filter(frame -> frame.blur ≤ threshold, selected)
end

# Derives the window length that yields roughly `num_frames` frames.
function window_for(frames::Vector{Frame}, num_frames::Int)
    duration = frames[end].time - frames[1].time
    if isnan(duration) || duration ≤ 0
        error("--num_frames needs timestamps the video does not carry, " *
            "use --window instead.")
    end
    # One extra window, so that the trailing partial one does not push the
    # count over the target.
    duration / num_frames * (num_frames + 1) / num_frames
end

function quantile_of(sorted::Vector{Float64}, q::Float64)
    isempty(sorted) && return NaN
    sorted[clamp(round(Int, q * (length(sorted) - 1)) + 1, 1, length(sorted))]
end

function report(frames::Vector{Frame}, selected::Vector{Frame})
    scores = sort!([frame.blur for frame in frames])
    println("Scored $(length(frames)) frames, blur score " *
        "min $(round(scores[1]; digits=2)), " *
        "median $(round(quantile_of(scores, 0.5); digits=2)), " *
        "max $(round(scores[end]; digits=2)).")

    if isempty(selected)
        @warn "No frames left, --blur_threshold is likely too low."
        return
    end
    kept = sort!([frame.blur for frame in selected])
    println("Keeping $(length(selected)) frames, blur score " *
        "min $(round(kept[1]; digits=2)), " *
        "median $(round(quantile_of(kept, 0.5); digits=2)), " *
        "max $(round(kept[end]; digits=2)).")
    return
end

# Writes out the selected frames. The whole selection goes through a single
# decode: seeking to each frame separately would be slower on long videos and
# unreliable on the ones without exact seek points.
function extract_frames(
    args::Args, ffmpeg_command::AbstractString, selected::Vector{Frame},
)
    input_path = joinpath(args.output_path, "input")
    mkpath(input_path)

    expression = join(("eq(n\\,$(frame.index))" for frame in selected), "+")
    graph = "select='$expression'"
    args.max_width > 0 &&
        (graph *= ",scale='min(iw,$(args.max_width))':-2:flags=lanczos")

    # The expression outgrows the command line on long videos, so the filter
    # graph is passed as a file.
    graph_path = joinpath(mktempdir(), "select.txt")
    write(graph_path, graph)

    quality = args.format == "jpg" ? ["-q:v", string(args.quality)] : String[]
    pattern = joinpath(input_path, "%05d.$(args.format)")

    println("Writing $(length(selected)) frames to $input_path...")
    run_or_exit(`$ffmpeg_command -hide_banner -loglevel warning -nostdin -y
        $(trim_options(args)) -i $(args.video)
        -map 0:v:0 -an -sn -dn
        -filter_script:v $graph_path -fps_mode passthrough
        $quality -start_number 1 $pattern`, "Frame extraction")

    written = count(f -> endswith(f, ".$(args.format)"), readdir(input_path))
    written == length(selected) || @warn(
        "Expected $(length(selected)) frames, found $written in $input_path.")
    return input_path
end

function main(argv::Vector{String})
    args = parse_args(argv)
    ffmpeg_command = isempty(args.ffmpeg_executable) ?
        "ffmpeg" : args.ffmpeg_executable
    check_ffmpeg(ffmpeg_command)

    frames = score_frames(args, ffmpeg_command)
    window = args.num_frames > 0 ?
        window_for(frames, args.num_frames) : args.window
    selected = select_frames(frames, window, args.blur_threshold)
    report(frames, selected)
    isempty(selected) && exit(1)

    if args.dry_run
        println("Dry run, no frames written.")
        return
    end

    extract_frames(args, ffmpeg_command, selected)
    println("Done. Reconstruct with:")
    println("  julia $(joinpath(@__DIR__, "gs-convert.jl")) -s $(args.output_path)")
    return
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
