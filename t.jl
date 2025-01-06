using FFTW
using CUDA
using Flux.NNlib
using Zygote

function main()
    n_fft = 1024
    hop_length = n_fft ÷ 4

    for i in 1:10
        @show i
        x = CuArray(rand(Float32, 8192, 1, 1))
        Zygote.gradient(x) do x
            abs(sum(NNlib.stft(x; window=nothing, n_fft, hop_length)))
        end
        GC.gc(false)
        GC.gc(true)
    end

    @show CUDA.CUFFT.idle_handles.idle_handles |> length
    @show CUDA.CUFFT.idle_handles.active_handles |> length
    return
end
main()
