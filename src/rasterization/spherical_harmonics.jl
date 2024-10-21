@kernel cpu=false function spherical_harmonics!(
    # Output.
    rgbs::AbstractVector{SVector{3, Float32}},
    clamped::AbstractVector{SVector{3, Bool}},
    # Inputs.
    radii::AbstractVector{Int32},
    means::AbstractVector{SVector{3, Float32}},
    camera_position::SVector{3, Float32},
    spherical_harmonics::AbstractMatrix{SVector{3, Float32}},
    degree,
)
    i = @index(Global)
    radii[i] > 0 || return

    mean = means[i]
    rgbs[i], clamped[i] = compute_colors_from_sh(
        mean, camera_position, @view(spherical_harmonics[:, i]), degree)
end

@kernel cpu=false function ∇spherical_harmonics!(
    # Output.
    vshs::AbstractMatrix{SVector{3, Float32}},
    vmeans::AbstractVector{SVector{3, Float32}},
    # Input.
    means::AbstractVector{SVector{3, Float32}},
    shs::AbstractMatrix{SVector{3, Float32}},
    clamped::AbstractVector{SVector{3, Bool}},
    vcolors::AbstractVector{SVector{3, Float32}},
    camera_position::SVector{3, Float32},
    sh_degree,
)
    i = @index(Global)
    vmean = ∇color_from_sh!(
        @view(vshs[:, i]),
        means[i], camera_position, @view(shs[:, i]),
        sh_degree, clamped[i], vcolors[i])
    vmeans[i] += vmean
end

# Convert spherical harmonics coefficients of each Gaussian to a RGB color.
@inbounds @inline function compute_colors_from_sh(
    point::SVector{3, Float32}, camera_position::SVector{3, Float32},
    shs::AbstractVector{SVector{3, Float32}}, ::Val{degree}
) where degree
    res = SH0 * shs[1]
    if degree > 0
        dir = normalize(point - camera_position)
        x, y, z = dir
        res = res - SH1 * y * shs[2] + SH1 * z * shs[3] - SH1 * x * shs[4]
        if degree > 1
            x², y², z² = x^2, y^2, z^2
            xy, xz, yz = x * y, x * z, y * z
            res = res +
                SH2C1 * xy * shs[5] +
                SH2C2 * yz * shs[6] +
                SH2C3 * (2f0 * z² - x² - y²) * shs[7] +
                SH2C4 * xz * shs[8] +
                SH2C5 * (x² - y²) * shs[9]

            if degree > 2
                res = res +
                    SH3C1 * y * (3f0 * x² - y²) * shs[10] +
                    SH3C2 * xy * z * shs[11] +
                    SH3C3 * y * (4f0 * z² - x² - y²) * shs[12] +
                    SH3C4 * z * (2f0 * z² - 3f0 * x² - 3f0 * y²) * shs[13] +
                    SH3C5 * x * (4f0 * z² - x² - y²) * shs[14] +
                    SH3C6 * z * (x² - y²) * shs[15] +
                    SH3C7 * x * (x² - 3f0 * y²) * shs[16]
            end
        end
    end
    res = res .+ 0.5f0 .+ eps(Float32) # Add for stability.
    return max.(0f0, res), (res .< 0f0)
end

@inbounds function ∇color_from_sh!(
    # Outputs.
    ∂L∂shs::AbstractVector{SVector{3, Float32}},
    # Inputs.
    point::SVector{3, Float32},
    camera_position::SVector{3, Float32},
    shs::AbstractVector{SVector{3, Float32}}, ::Val{degree},
    clamped::SVector{3, Bool},
    ∂L∂color::SVector{3, Float32},
) where degree
    dir_orig = point - camera_position
    dir = normalize(dir_orig)

    # If clamped - gradient is 0.
    ∂L∂color = ∂L∂color .* (1f0 .- clamped)
    ∂color∂x = zeros(SVector{3, Float32})
    ∂color∂y = zeros(SVector{3, Float32})
    ∂color∂z = zeros(SVector{3, Float32})

    ∂L∂shs[1] = SH0 * ∂L∂color
    if degree > 0
        x, y, z = dir
        ∂L∂shs[2] = -SH1 * y * ∂L∂color
        ∂L∂shs[3] =  SH1 * z * ∂L∂color
        ∂L∂shs[4] = -SH1 * x * ∂L∂color

        ∂color∂x = -SH1 * shs[4]
        ∂color∂y = -SH1 * shs[2]
        ∂color∂z =  SH1 * shs[3]
        if degree > 1
            x², y², z² = x^2, y^2, z^2
            xy, xz, yz = x * y, x * z, y * z

            ∂L∂shs[5] = SH2C1 * xy * ∂L∂color
            ∂L∂shs[6] = SH2C2 * yz * ∂L∂color
            ∂L∂shs[7] = SH2C3 * (2f0 * z² - x² - y²) * ∂L∂color
            ∂L∂shs[8] = SH2C4 * xz * ∂L∂color
            ∂L∂shs[9] = SH2C5 * (x² - y²) * ∂L∂color

            ∂color∂x = ∂color∂x +
                SH2C1 * y * shs[5] +
                SH2C3 * 2f0 * -x * shs[7] +
                SH2C4 * z * shs[8] +
                SH2C5 * 2f0 * x * shs[9]
            ∂color∂y = ∂color∂y +
                SH2C1 * x * shs[5] +
                SH2C2 * z * shs[6] +
                SH2C3 * 2f0 * -y * shs[7] +
                SH2C5 * 2f0 * -y * shs[9]
            ∂color∂z = ∂color∂z +
                SH2C2 * y * shs[6] +
                SH2C3 * 4f0 * z * shs[7] +
                SH2C4 * x * shs[8]
            if degree > 2
                ∂L∂shs[10] = SH3C1 * y * (3f0 * x² - y²) * ∂L∂color
                ∂L∂shs[11] = SH3C2 * xy * z * ∂L∂color
                ∂L∂shs[12] = SH3C3 * y * (4f0 * z² - x² - y²) * ∂L∂color
                ∂L∂shs[13] = SH3C4 * z * (2f0 * z² - 3f0 * x² - 3f0 * y²) * ∂L∂color
                ∂L∂shs[14] = SH3C5 * x * (4f0 * z² - x² - y²) * ∂L∂color
                ∂L∂shs[15] = SH3C6 * z * (x² - y²) * ∂L∂color
                ∂L∂shs[16] = SH3C7 * x * (x² - 3f0 * y²) * ∂L∂color

                ∂color∂x = ∂color∂x +
                    SH3C1 * shs[10] * 3f0 * 2f0 * xy +
                    SH3C2 * shs[11] * yz +
                    SH3C3 * shs[12] * -2f0 * xy +
                    SH3C4 * shs[13] * -3f0 * 2f0 * xz +
                    SH3C5 * shs[14] * (-3f0 * x² + 4f0 * z² - y²) +
                    SH3C6 * shs[15] * 2f0 * xz +
                    SH3C7 * shs[16] * 3f0 * (x² - y²)
                ∂color∂y = ∂color∂y +
                    SH3C1 * shs[10] * 3f0 * (x² - y²) +
                    SH3C2 * shs[11] * xz +
                    SH3C3 * shs[12] * (-3f0 * y² + 4f0 * z² - x²) +
                    SH3C4 * shs[13] * -3f0 * 2f0 * yz +
                    SH3C5 * shs[14] * -2f0 * xy +
                    SH3C6 * shs[15] * -2f0 * yz +
                    SH3C7 * shs[16] * -3f0 * 2f0 * xy
                ∂color∂z = ∂color∂z +
                    SH3C2 * shs[11] * xy +
                    SH3C3 * shs[12] * 4f0 * 2f0 * yz +
                    SH3C4 * shs[13] * 3f0 * (2f0 * z² - x² - y²) +
                    SH3C5 * shs[14] * 4f0 * 2f0 * xz +
                    SH3C6 * shs[15] * (x² - y²)
            end
        end
    end

    # The view direction is an input to the computation.
    # View direction is influenced by the Gaussian's mean,
    # so SHs gradients must propagate back into 3D position.
    ∂L∂dir = SVector{3, Float32}(
        ∂color∂x ⋅ ∂L∂color, ∂color∂y ⋅ ∂L∂color, ∂color∂z ⋅ ∂L∂color)

    # Account for normalization.
    return ∇normalize(dir_orig, ∂L∂dir)
end

@inbounds function ∇normalize(dir::SVector{3, Float32}, ∂L∂dir::SVector{3, Float32})
    s² = sum(abs2, dir)
    inv_s = 1f0 / √(s²^3)
    SVector{3, Float32}(
        ((s² - dir[1]^2) * ∂L∂dir[1] - dir[2] * dir[1] * ∂L∂dir[2] - dir[3] * dir[1] * ∂L∂dir[3]) * inv_s,
        (-dir[1] * dir[2] * ∂L∂dir[1] + (s² - dir[2]^2) * ∂L∂dir[2] - dir[3] * dir[2] * ∂L∂dir[3]) * inv_s,
        (-dir[1] * dir[3] * ∂L∂dir[1] - dir[2] * dir[3] * ∂L∂dir[2] + (s² - dir[3]^2) * ∂L∂dir[3]) * inv_s)
end
