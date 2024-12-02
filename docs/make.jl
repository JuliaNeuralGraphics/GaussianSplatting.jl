using Documenter
using GaussianSplatting

function main()
    ci = get(ENV, "CI", "") == "true"

    @info "Generating Documenter site"
    makedocs(;
        modules=[GaussianSplatting],
        sitename="GaussianSplatting.jl",
        format=Documenter.HTML(;
            prettyurls=ci,
            assets=["assets/favicon.ico"],
            analytics="UA-154489943-2",
        ),
        warnonly=[:missing_docs],
        pages=[
            "Home" => "index.md",
            "API" => "api.md",
            # "Examples" => "examples.md",
        ],
    )
    if ci
        @info "Deploying to GitHub"
        deploydocs(;
            repo="github.com/JuliaNeuralGraphics/GaussianSplatting.jl.git",
            push_preview=true,
        )
    end
end

isinteractive() || main()
