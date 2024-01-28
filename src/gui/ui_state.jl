Base.@kwdef mutable struct UIState
    train::Ref{Bool} = Ref(false)
    render::Ref{Bool} = Ref(true)
    loss::Float32 = 0f0

    draw_cameras::Ref{Bool} = Ref(false)
    selected_view::Ref{Int32} = Ref{Int32}(0)

    save_directory_path::Vector{UInt8} = Vector{UInt8}("\0"^512)
    state_file::Vector{UInt8} = Vector{UInt8}("\0"^512)

    sh_degree::Ref{Int32} = Ref{Int32}(-1) # -1 means use value from the model

    selected_mode::Ref{Int32} = Ref{Int32}(0)
    render_modes::Vector{String} = ["Color", "Depth", "Uncertainty"]
end
