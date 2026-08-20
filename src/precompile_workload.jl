"""
    precompile_workload(kab; mode::Symbol = :rgbd, sh_degrees = 0:0)

Minimal workload to precompile GPU kernels.
"""
function precompile_workload(kab; mode::Symbol = :rgbd, sh_degrees = 0:0)
    width, height = 64, 48
    camera = Camera(; fx=100f0, fy=100f0, width, height)

    # A grid of near-opaque Gaussians at `z = 3`.
    xs = range(-0.6f0, 0.6f0; length=8)
    points = Float32[p[i] for i in 1:3, p in vec([(x, y, 3f0) for x in xs, y in xs])]
    n_points = size(points, 2)
    colors = fill(0.5f0, 3, n_points)
    scales = repeat(Float32[log(0.2f0), log(0.2f0), log(0.2f0)], 1, n_points)

    gaussians = GaussianModel(kab, points, colors, scales; max_sh_degree=3, isotropic=false)
    fill!(gaussians.opacities, 5f0) # Make gaussians opaque.

    rast = GaussianRasterizer(kab, camera; mode)
    try
        # Multiply by `weights` to materialize cotangent from `Fill` to actual array,
        # otherwise `∇rasterize` will fail.
        weights = KA.ones(kab, Float32, (n_color_features(mode), width, height))
        try
            for sh_degree in sh_degrees
                # Precompile inference branch (used in GUI & eval).
                rast(
                    gaussians.points, gaussians.opacities, gaussians.scales,
                    gaussians.rotations, gaussians.features_dc,
                    gaussians.features_rest; camera, sh_degree)

                # Precompile forward+backward pass of the rasterizer.
                Zygote.withgradient(
                    gaussians.points, gaussians.features_dc, gaussians.features_rest,
                    gaussians.opacities, gaussians.scales, gaussians.rotations,
                ) do means_3d, features_dc, features_rest, opacities, scales, rotations
                    features = rast(
                        means_3d, opacities, scales, rotations,
                        features_dc, features_rest; camera, sh_degree)
                    sum(features .* weights)
                end
            end
        finally
            KA.unsafe_free!(weights)
        end
    finally
        KA.unsafe_free!(rast)
        finalize(rast)
        KA.unsafe_free!(gaussians)
    end
    KA.synchronize(kab)
    return
end
