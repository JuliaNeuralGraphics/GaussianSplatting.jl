using GaussianSplatting
using Statistics
using GaussianSplatting.Adapt
using GaussianSplatting.StaticArrays
using GaussianSplatting.FileIO
using GaussianSplatting.ImageCore
using GaussianSplatting.ImageIO
using GaussianSplatting.VideoIO
using GaussianSplatting.Zygote

import GaussianSplatting as GSP

function main(dataset_path::String, state_path::String; scale::Int)
    kab = GSP.gpu_backend()
    @info "Using `$kab` GPU backend."

    dataset = GSP.ColmapDataset(kab, dataset_path; scale,
        train_test_split=1, permute=false)
    camera = dataset.train_cameras[1]
    target_camera = deepcopy(camera)

    # Shift camera by 0.3 along X-axis and attempt to recover original position.
    GSP.shift!(camera, SVector{3, Float32}(0.5, 0.0f0, 0.0f0))

    @info "Dataset resolution: $(Int.(camera.intrinsics.resolution))"

    n_steps = 500
    lr_start, lr_end = 1f-3, 1f-4
    copt = GSP.CameraOpt(camera; lr=lr_start)
    lr_scheduler = GSP.lr_exp_scheduler(lr_start, lr_end, n_steps)

    opt_params = GSP.OptimizationParams()
    gaussians = GSP.GaussianModel(
        dataset.points, dataset.colors, dataset.scales)
    rasterizer = GSP.GaussianRasterizer(kab, camera;
        antialias=false, fused=true, mode=:rgb)
    trainer = GSP.Trainer(rasterizer, gaussians, dataset, opt_params)
    GSP.load_state!(trainer, state_path)

    image = rasterizer(
        gaussians.points, gaussians.opacities, gaussians.scales,
        gaussians.rotations, gaussians.features_dc, gaussians.features_rest;
        camera=target_camera, sh_degree=gaussians.sh_degree)
    target_rgb_image = RGB{N0f8}.(GSP.to_image(image))

    res = GSP.resolution(camera)
    writer = open_video_out(
        "./out.mp4", zeros(RGB{N0f8}, res.height, res.width * 2);
        framerate=60, target_pix_fmt=VideoIO.AV_PIX_FMT_YUV420P)

    target_image = GSP.get_image(trainer, 1, :train)
    for i in 1:n_steps
        # copt.opt.lr = lr_scheduler(i)

        loss, ∇ = Zygote.withgradient(copt.drot, copt.dt) do drot, dt
            R, t = GSP.pose_delta(copt.R_w2c, copt.t_w2c, drot, dt; copt.id)
            Rd = adapt(kab, R)
            td = adapt(kab, t)

            image = rasterizer(
                gaussians.points, gaussians.opacities, gaussians.scales,
                gaussians.rotations, gaussians.features_dc, gaussians.features_rest,
                Rd, td; camera, sh_degree=gaussians.sh_degree)

            # From (c, w, h) to (w, h, c, 1).
            image_tmp = permutedims(image, (2, 3, 1))
            image_eval = reshape(image_tmp, size(image_tmp)..., 1)
            mean(abs.(image_eval .- target_image))
        end
        GSP.apply!(copt, ∇)

        if i % 1 == 0
            @show i, loss
            image = rasterizer(
                gaussians.points, gaussians.opacities, gaussians.scales,
                gaussians.rotations, gaussians.features_dc, gaussians.features_rest;
                camera, sh_degree=gaussians.sh_degree)
            rgb_image = GSP.to_image(image)
            write(writer, hcat(RGB{N0f8}.(rgb_image), target_rgb_image))
        end
    end
    close_video_out!(writer)
    return
end
