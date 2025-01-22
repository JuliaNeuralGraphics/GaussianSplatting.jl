mse(x, y) = mean((x .- y).^2)

psnr(x, y) = 20f0 * log10(1f0 / sqrt(mse(x, y)))

struct SSIM{W <: Flux.Conv}
    window::W
    c1::Float32
    c2::Float32
end

function SSIM(kab; channels::Int = 3, σ::Float32 = 1.5f0, window_size::Int = 11)
    w = ImageFiltering.KernelFactors.gaussian(σ, window_size)
    w2d = reshape(reshape(w, :, 1) * reshape(w, 1, :), window_size, window_size, 1)
    window = reshape(
        repeat(w2d, 1, 1, channels),
        window_size, window_size, 1, channels)

    conv = Flux.Conv(
        (window_size, window_size), channels => channels;
        pad=(window_size ÷ 2, window_size ÷ 2),
        groups=channels, bias=false)
    copy!(conv.weight, window)

    SSIM(kab != CPU() ? Flux.gpu(conv) : conv, 0.01f0^2, 0.03f0^2)
end

function (ssim::SSIM)(x::T, ref::T) where T
    μ₁, μ₂ = ssim.window(x), ssim.window(ref)
    μ₁², μ₂² = μ₁.^2, μ₂.^2
    μ₁₂ = μ₁ .* μ₂

    σ₁² = ssim.window(x.^2) .- μ₁²
    σ₂² = ssim.window(ref.^2) .- μ₂²
    σ₁₂ = ssim.window(x .* ref) .- μ₁₂

    l = ((2f0 .* μ₁₂ .+ ssim.c1) .* (2f0 .* σ₁₂ .+ ssim.c2)) ./
        ((μ₁² .+ μ₂² .+ ssim.c1) .* (σ₁² .+ σ₂² .+ ssim.c2))
    return mean(l)
end
