using AMDGPU
using Adapt
using Statistics
using Zygote
using GaussianSplatting
import KernelAbstractions as KA

const GS = GaussianSplatting

function main(kab)
    w, h, n_images = 64, 48, 4
    idx = 2 # Current "train view".

    opt_params = GS.OptimizationParams(;
        use_bilateral_grid=true, bilateral_grid_size=(8, 8, 4))
    bgrid = GS.BilateralGrid(kab, n_images, opt_params)
    image = adapt(kab, rand(Float32, 3, w, h))
    @info "Grids: $(size(bgrid.grids)), image: $(size(image))"

    # 1) Identity grids leave the image unchanged.
    out = GS.bilateral_slice(image, bgrid.grids[:, :, :, :, idx])
    @assert Array(out) ≈ Array(image)
    @info "Identity slice: max abs err = $(maximum(abs.(Array(out) .- Array(image))))"

    # 2) Identity grids have no variation.
    @assert GS.tv_loss(bgrid.grids) ≈ 0f0
    @info "TV loss on identity grids: $(GS.tv_loss(bgrid.grids))"

    # 3) Gradients through slice + TV, as in `step!`.
    target = adapt(kab, rand(Float32, 3, w, h))
    loss, ∇ = Zygote.withgradient(bgrid.grids, image) do grids, img
        corrected = GS.bilateral_slice(img, grids[:, :, :, :, idx])
        mean(abs.(corrected .- target)) +
            opt_params.tv_loss_weight * GS.tv_loss(grids)
    end
    ∇grids, ∇image = ∇
    @assert size(∇grids) == size(bgrid.grids)
    @assert size(∇image) == size(image)
    @assert isfinite(sum(∇grids)) && isfinite(sum(∇image))
    # Identity grids are constant along guidance, so the z-path cancels &
    # only the sliced view receives a photometric gradient.
    other = [i for i in 1:n_images if i != idx]
    @assert maximum(abs.(Array(∇grids[:, :, :, :, other]))) ≈ 0f0
    @info "Loss: $loss, ‖∇grids‖∞ = $(maximum(abs.(Array(∇grids)))), " *
        "‖∇image‖∞ = $(maximum(abs.(Array(∇image))))"

    # # 4) Short optimization: the grid alone absorbs a global
    # # exposure / white-balance shift of the "render" towards the target.
    # shift = adapt(kab, Float32[0.7f0, 0.9f0, 1.1f0])
    # target = clamp.(reshape(shift, 3, 1, 1) .* image .+ 0.05f0, 0f0, 1f0)
    # l0 = nothing
    # for i in 1:200
    #     bgrid.optimizer.lr = bgrid.scheduler(1000 + i) # Skip warmup.
    #     l, ∇ = Zygote.withgradient(bgrid.grids) do grids
    #         corrected = GS.bilateral_slice(image, grids[:, :, :, :, idx])
    #         mean(abs.(corrected .- target)) +
    #             opt_params.tv_loss_weight * GS.tv_loss(grids)
    #     end
    #     NU.step!(bgrid.optimizer, bgrid.grids, ∇[1]; dispose=false)
    #     l0 = l0 ≡ nothing ? l : l0
    #     (i % 50 == 0) && @info "  fit step $i: loss = $l"
    # end
    # @info "Exposure fit: initial loss = $l0"

    KA.synchronize(kab)
    @info "OK"
    return
end

main(ROCBackend())
