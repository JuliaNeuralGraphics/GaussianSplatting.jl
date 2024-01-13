function is_valid_backend(name::String)
    (name == "AMDGPU" || name == "CUDA") && return name
    error("""
    Invalid `backend` specified: $name.
    Supported values are: AMDGPU, CUDA.
    """)
end

backend_name() = is_valid_backend(@load_preference("backend", "AMDGPU"))

function set_backend!(name::String)::Bool
    backend_name() == name && return false

    is_valid_backend(name)
    @set_preferences!("backend" => name)
    @info """Successfully changed backend to: $name.
    Restart your Julia session for the changes to take effect."""
    return true
end

const BACKEND_NAME::String = backend_name()

@static if BACKEND_NAME == "AMDGPU"
    using AMDGPU
    const Backend::ROCBackend = ROCBackend()

    get_module(::ROCBackend) = AMDGPU

    synchronize_count(predicate::Cint) = AMDGPU.sync_workgroup_count(predicate)
elseif BACKEND_NAME == "CUDA"
    using CUDA
    const Backend::CUDABackend = CUDABackend()

    get_module(::CUDABackend) = CUDA

    synchronize_count(predicate::Cint) = CUDA.sync_threads_count(predicate)
end
