mutable struct CameraOpt{R <: AbstractMatrix, D <: AbstractVector}
    camera::Camera

    R_w2c::R
    t_w2c::D

    id::D
    # Trainable deltas.
    drot::D # 6 - elements for rotation.
    dt::D # 3 - elements for translation.
end

function CameraOpt(kab, camera::Camera)
    R_w2c = adapt(kab, camera.w2c[1:3, 1:3])
    t_w2c = adapt(kab, camera.w2c[1:3, 4])

    id = adapt(kab, Float32[1, 0, 0, 0, 1, 0])
    drot = KA.zeros(kab, Float32, (6,))
    dt = KA.zeros(kab, Float32, (3,))
    CameraOpt(camera, R_w2c, t_w2c, id, drot, dt)
end

function pose_delta(R_w2c, t_w2c, drot, dt; id)
    dR = rotation_6d_to_matrix(drot .+ id)
    new_R = dR * R_w2c
    new_t = dR * t_w2c .+ dt
    return new_R, new_t
end

# TODO apply!(copt::CameraOpt, ∇)
# - update `R_w2c`, `t_w2c` & `camera` afterwards
