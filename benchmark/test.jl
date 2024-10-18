using GaussianSplatting
using GaussianSplatting.ImageCore
using GaussianSplatting.ImageIO
using GaussianSplatting.FileIO

function main(dataset_path::String, scale::Int = 1)
    kab = GaussianSplatting.gpu_backend()

    dataset = GaussianSplatting.ColmapDataset(kab, dataset_path; scale)
    opt_params = GaussianSplatting.OptimizationParams()
    gaussians = GaussianSplatting.GaussianModel(
        dataset.points, dataset.colors, dataset.scales)
    rasterizer = GaussianSplatting.GaussianRasterizer(kab,
        dataset.cameras[1]; auxiliary=false)
    trainer = GaussianSplatting.Trainer(rasterizer, gaussians, dataset, opt_params)

    cam_idx = rand(1:length(trainer.dataset.cameras))
    camera = trainer.dataset.cameras[cam_idx]

    shs = isempty(gaussians.features_rest) ?
        gaussians.features_dc :
        hcat(gaussians.features_dc, gaussians.features_rest)
    rasterizer(
        gaussians.points, gaussians.opacities, gaussians.scales,
        gaussians.rotations, shs; camera, sh_degree=gaussians.sh_degree,
        covisibility=nothing)
    save("test.png", GaussianSplatting.to_image(rasterizer))

    return
end
main("/home/pxlth/Downloads/360_v2/bicycle", 4)
