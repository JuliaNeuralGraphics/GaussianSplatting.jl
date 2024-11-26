mutable struct CameraOpt{O}
    camera::Camera

    R_w2c::Matrix{Float32}
    t_w2c::Vector{Float32}

    id::Vector{Float32}
    # Trainable deltas.
    drot::Vector{Float32} # 6 - elements for rotation.
    dt::Vector{Float32} # 3 - elements for translation.

    opt::O
end

function CameraOpt(camera::Camera; lr::Float32)
    R_w2c = Array(camera.w2c[1:3, 1:3])
    t_w2c = Array(camera.w2c[1:3, 4])

    id = Float32[1, 0, 0, 0, 1, 0]
    drot = zeros(Float32, 6)
    dt = zeros(Float32, 3)
    opt = NU.Adam(CPU(), (drot, dt); lr)
    CameraOpt(camera, R_w2c, t_w2c, id, drot, dt, opt)
end

function pose_delta(R_w2c, t_w2c, drot, dt; id)
    dR = rotation_6d_to_matrix(drot .+ id)
    new_R = R_w2c * dR
    new_t = R_w2c * dt .+ t_w2c
    return new_R, new_t
end

function apply!(copt::CameraOpt, ∇; dispose::Bool = false)
    NU.step!(copt.opt, (copt.drot, copt.dt), ∇; dispose)
    update_camera!(copt)

    NU.reset!(copt.opt)
    fill!(copt.drot, 0f0)
    fill!(copt.dt, 0f0)
    return
end

function update_camera!(copt::CameraOpt; δ::Float32 = 1f-3)
    new_R, new_t = pose_delta(copt.R_w2c, copt.t_w2c, copt.drot, copt.dt; copt.id)
    copt.R_w2c .= new_R
    copt.t_w2c .= new_t

    w2c = MMatrix(copt.camera.w2c)
    w2c[1:3, 1:3] .= new_R
    w2c[1:3, 4] .= new_t
    set_c2w!(copt.camera, inv(w2c))
    return
end
