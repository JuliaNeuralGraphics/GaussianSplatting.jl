const BLOCK_X = 16
const BLOCK_Y = 16
const HALO = 5

const SHARED_X = BLOCK_X + 2 * HALO # 26
const SHARED_Y = BLOCK_Y + 2 * HALO # 26

const CONV_X = BLOCK_X  # 16
const CONV_Y = SHARED_Y # 26

# Pre-computed 11-element Gaussian kernel.
const GAUSS = (
    0.001028380123898387f0,
    0.0075987582094967365f0,
    0.036000773310661316f0,
    0.10936068743467331f0,
    0.21300552785396576f0,
    0.26601171493530273f0,
    0.21300552785396576f0,
    0.10936068743467331f0,
    0.036000773310661316f0,
    0.0075987582094967365f0,
    0.001028380123898387f0,
)

# Safe pixel fetch with zero padding for out-of-bounds access.
@inline function get_pix_value(img, b::Int, c::Int, y::Int, x::Int)
    W, H = size(img, 1), size(img, 2)
    (x < 1 || x > W || y < 1 || y > H) && return 0f0
    return @inbounds img[x, y, c, b]
end

# Forward kernel: Fused SSIM computation.
@kernel cpu=false unsafe_indices=true inbounds=true function _fused_ssim!(
    ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12,
    @Const(img), @Const(ref),
    C1::Float32, C2::Float32, train::Bool,
)
    bx, by, bz = @index(Group, NTuple)
    tx, ty = @index(Local, NTuple)

    W, H, CH, B = size(img)
    pix_x = (bx - 1) * BLOCK_X + tx
    pix_y = (by - 1) * BLOCK_Y + ty

    # Shared memory for the tile (img, ref).
    sTile = @localmem Float32 (2, SHARED_X, SHARED_Y)
    # After horizontal pass: (sumX, sumX², sumY, sumY², sumXY).
    xconv = @localmem Float32 (5, CONV_X, CONV_Y)

    # Loop over channels.
    for c in 1:CH
        # 1) Load (img, ref) tile + halo into shared memory.
        tile_size = SHARED_Y * SHARED_X
        threads = BLOCK_X * BLOCK_Y
        steps = cld(tile_size, threads)

        tile_start_y = (by - 1) * BLOCK_Y + 1
        tile_start_x = (bx - 1) * BLOCK_X + 1

        tid = (ty - 1) * BLOCK_X + tx  # 1-based thread rank.

        for s in 0:(steps - 1)
            flat_id = s * threads + tid
            if flat_id ≤ tile_size
                local_y = cld(flat_id, SHARED_X)
                local_x = mod1(flat_id, SHARED_X)

                # Global coordinates with halo offset.
                gy = tile_start_y + local_y - 1 - HALO
                gx = tile_start_x + local_x - 1 - HALO

                X = get_pix_value(img, bz, c, gy, gx)
                Y = get_pix_value(ref, bz, c, gy, gx)

                sTile[1, local_x, local_y] = X
                sTile[2, local_x, local_y] = Y
            end
        end
        @synchronize

        # 2) Horizontal convolution (11×1) in shared memory.
        ly = ty
        lx = tx + HALO  # Skip left halo.

        sumX = 0f0
        sumX2 = 0f0
        sumY = 0f0
        sumY2 = 0f0
        sumXY = 0f0

        # Symmetric pairs around center.
        @unroll for d in 1:HALO
            w = GAUSS[HALO + 1 - d]
            Xleft = sTile[1, lx - d, ly]
            Yleft = sTile[2, lx - d, ly]
            Xright = sTile[1, lx + d, ly]
            Yright = sTile[2, lx + d, ly]

            sumX += (Xleft + Xright) * w
            sumX2 += (Xleft * Xleft + Xright * Xright) * w
            sumY += (Yleft + Yright) * w
            sumY2 += (Yleft * Yleft + Yright * Yright) * w
            sumXY += (Xleft * Yleft + Xright * Yright) * w
        end

        # Center.
        centerX = sTile[1, lx, ly]
        centerY = sTile[2, lx, ly]
        wc = GAUSS[HALO + 1]
        sumX += centerX * wc
        sumX2 += centerX * centerX * wc
        sumY += centerY * wc
        sumY2 += centerY * centerY * wc
        sumXY += centerX * centerY * wc

        # Write partial sums.
        xconv[1, tx, ly] = sumX
        xconv[2, tx, ly] = sumX2
        xconv[3, tx, ly] = sumY
        xconv[4, tx, ly] = sumY2
        xconv[5, tx, ly] = sumXY

        # Handle second row (threads handle 2 rows to cover CONV_Y = 26).
        ly2 = ly + BLOCK_Y
        if ly2 ≤ CONV_Y
            sumX = 0f0
            sumX2 = 0f0
            sumY = 0f0
            sumY2 = 0f0
            sumXY = 0f0

            @unroll for d in 1:HALO
                w = GAUSS[HALO + 1 - d]
                Xleft = sTile[1, lx - d, ly2]
                Yleft = sTile[2, lx - d, ly2]
                Xright = sTile[1, lx + d, ly2]
                Yright = sTile[2, lx + d, ly2]

                sumX += (Xleft + Xright) * w
                sumX2 += (Xleft * Xleft + Xright * Xright) * w
                sumY += (Yleft + Yright) * w
                sumY2 += (Yleft * Yleft + Yright * Yright) * w
                sumXY += (Xleft * Yleft + Xright * Yright) * w
            end

            cx = sTile[1, lx, ly2]
            cy = sTile[2, lx, ly2]
            sumX += cx * wc
            sumX2 += cx * cx * wc
            sumY += cy * wc
            sumY2 += cy * cy * wc
            sumXY += cx * cy * wc

            xconv[1, tx, ly2] = sumX
            xconv[2, tx, ly2] = sumX2
            xconv[3, tx, ly2] = sumY
            xconv[4, tx, ly2] = sumY2
            xconv[5, tx, ly2] = sumXY
        end
        @synchronize

        # 3) Vertical convolution (1×11) + final SSIM.
        ly_v = ty + HALO
        lx_v = tx

        out0 = 0f0
        out1 = 0f0
        out2 = 0f0
        out3 = 0f0
        out4 = 0f0

        @unroll for d in 1:HALO
            w = GAUSS[HALO + 1 - d]
            top0 = xconv[1, lx_v, ly_v - d]
            top1 = xconv[2, lx_v, ly_v - d]
            top2 = xconv[3, lx_v, ly_v - d]
            top3 = xconv[4, lx_v, ly_v - d]
            top4 = xconv[5, lx_v, ly_v - d]

            bot0 = xconv[1, lx_v, ly_v + d]
            bot1 = xconv[2, lx_v, ly_v + d]
            bot2 = xconv[3, lx_v, ly_v + d]
            bot3 = xconv[4, lx_v, ly_v + d]
            bot4 = xconv[5, lx_v, ly_v + d]

            out0 += (top0 + bot0) * w
            out1 += (top1 + bot1) * w
            out2 += (top2 + bot2) * w
            out3 += (top3 + bot3) * w
            out4 += (top4 + bot4) * w
        end

        # Center.
        wC = GAUSS[HALO + 1]
        out0 += xconv[1, lx_v, ly_v] * wC
        out1 += xconv[2, lx_v, ly_v] * wC
        out2 += xconv[3, lx_v, ly_v] * wC
        out3 += xconv[4, lx_v, ly_v] * wC
        out4 += xconv[5, lx_v, ly_v] * wC

        if pix_x ≤ W && pix_y ≤ H
            mu1 = out0
            mu2 = out2
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2

            sigma1_sq = out1 - mu1_sq
            sigma2_sq = out3 - mu2_sq
            sigma12 = out4 - mu1 * mu2

            A = mu1_sq + mu2_sq + C1
            B_val = sigma1_sq + sigma2_sq + C2
            C_val = 2f0 * mu1 * mu2 + C1
            D_val = 2f0 * sigma12 + C2

            val = (C_val * D_val) / (A * B_val)
            ssim_map[pix_x, pix_y, c, bz] = val

            if train
                # Partial derivatives for backpropagation.
                d_m_dmu1 = (
                    (mu2 * 2f0 * D_val) / (A * B_val) -
                    (mu2 * 2f0 * C_val) / (A * B_val) -
                    (mu1 * 2f0 * C_val * D_val) / (A * A * B_val) +
                    (mu1 * 2f0 * C_val * D_val) / (A * B_val * B_val)
                )
                d_m_dsigma1_sq = (-C_val * D_val) / (A * B_val * B_val)
                d_m_dsigma12 = (2f0 * C_val) / (A * B_val)

                dm_dmu1[pix_x, pix_y, c, bz] = d_m_dmu1
                dm_dsigma1_sq[pix_x, pix_y, c, bz] = d_m_dsigma1_sq
                dm_dsigma12[pix_x, pix_y, c, bz] = d_m_dsigma12
            end
        end
        @synchronize
    end
end

# Backward kernel: Compute dL/d(img) from partial derivatives and dL/dmap.
@kernel cpu=false unsafe_indices=true inbounds=true function _fused_ssim_bwd!(
    dL_dimg,
    @Const(img), @Const(ref), @Const(dL_dmap),
    @Const(dm_dmu1), @Const(dm_dsigma1_sq), @Const(dm_dsigma12),
)
    W, H, CH, B = size(img)

    bx, by, bz = @index(Group, NTuple)
    tx, ty = @index(Local, NTuple)

    pix_x = (bx - 1) * BLOCK_X + tx
    pix_y = (by - 1) * BLOCK_Y + ty

    # Shared memory for fused data: dm_dmu1*dL, dm_dsigma1_sq*dL, dm_dsigma12*dL.
    sData = @localmem Float32 (3, SHARED_X, SHARED_Y)
    sScratch = @localmem Float32 (3, CONV_X, CONV_Y)

    for c in 1:CH
        p1 = 0f0
        p2 = 0f0
        if pix_x ≤ W && pix_y ≤ H
            p1 = get_pix_value(img, bz, c, pix_y, pix_x)
            p2 = get_pix_value(ref, bz, c, pix_y, pix_x)
        end

        # 1) Load + fuse multiplication.
        start_y = (by - 1) * BLOCK_Y + 1
        start_x = (bx - 1) * BLOCK_X + 1

        tile_size = SHARED_Y * SHARED_X
        threads = BLOCK_X * BLOCK_Y
        steps = cld(tile_size, threads)
        tid = (ty - 1) * BLOCK_X + tx

        for s in 0:(steps - 1)
            flat_id = s * threads + tid
            if flat_id ≤ tile_size
                row = cld(flat_id, SHARED_X)
                col = mod1(flat_id, SHARED_X)

                gy = start_y + row - 1 - HALO
                gx = start_x + col - 1 - HALO

                chain = get_pix_value(dL_dmap, bz, c, gy, gx)
                vmu = get_pix_value(dm_dmu1, bz, c, gy, gx)
                vs1 = get_pix_value(dm_dsigma1_sq, bz, c, gy, gx)
                vs12 = get_pix_value(dm_dsigma12, bz, c, gy, gx)

                sData[1, col, row] = vmu * chain
                sData[2, col, row] = vs1 * chain
                sData[3, col, row] = vs12 * chain
            end
        end
        @synchronize

        # 2) Horizontal pass.
        ly = ty
        lx = tx + HALO

        for pass in 0:1
            yy = ly + pass * BLOCK_Y
            if yy ≤ CONV_Y
                accum0 = 0f0
                accum1 = 0f0
                accum2 = 0f0

                @unroll for d in 1:HALO
                    w = GAUSS[HALO + 1 - d]
                    left0 = sData[1, lx - d, yy]
                    left1 = sData[2, lx - d, yy]
                    left2 = sData[3, lx - d, yy]

                    right0 = sData[1, lx + d, yy]
                    right1 = sData[2, lx + d, yy]
                    right2 = sData[3, lx + d, yy]

                    accum0 += (left0 + right0) * w
                    accum1 += (left1 + right1) * w
                    accum2 += (left2 + right2) * w
                end

                # Center.
                wc = GAUSS[HALO + 1]
                accum0 += sData[1, lx, yy] * wc
                accum1 += sData[2, lx, yy] * wc
                accum2 += sData[3, lx, yy] * wc

                sScratch[1, tx, yy] = accum0
                sScratch[2, tx, yy] = accum1
                sScratch[3, tx, yy] = accum2
            end
        end
        @synchronize

        # 3) Vertical pass -> finalize dL/d(img).
        if pix_x ≤ W && pix_y ≤ H
            ly_v = ty + HALO
            lx_v = tx

            sum0 = 0f0
            sum1 = 0f0
            sum2 = 0f0

            @unroll for d in 1:HALO
                w = GAUSS[HALO + 1 - d]
                top0 = sScratch[1, lx_v, ly_v - d]
                top1 = sScratch[2, lx_v, ly_v - d]
                top2 = sScratch[3, lx_v, ly_v - d]

                bot0 = sScratch[1, lx_v, ly_v + d]
                bot1 = sScratch[2, lx_v, ly_v + d]
                bot2 = sScratch[3, lx_v, ly_v + d]

                sum0 += (top0 + bot0) * w
                sum1 += (top1 + bot1) * w
                sum2 += (top2 + bot2) * w
            end

            # Center.
            wc = GAUSS[HALO + 1]
            sum0 += sScratch[1, lx_v, ly_v] * wc
            sum1 += sScratch[2, lx_v, ly_v] * wc
            sum2 += sScratch[3, lx_v, ly_v] * wc

            # Final accumulation.
            dL_dpix = sum0 + 2f0 * p1 * sum1 + p2 * sum2
            dL_dimg[pix_x, pix_y, c, bz] = dL_dpix
        end
        @synchronize
    end
end

function _fused_ssim(
    img::T; ref::T, C1::Float32 = 0.01f0^2, C2::Float32 = 0.03f0^2, train::Bool,
) where T <: AbstractArray{Float32, 4}
    W, H, CH, B = size(img)
    kab = get_backend(img)

    ssim_map = KA.zeros(kab, Float32, W, H, CH, B)
    dm_dmu1 = train ? KA.zeros(kab, Float32, W, H, CH, B) : KA.zeros(kab, Float32, 0, 0, 0, 0)
    dm_dsigma1_sq = train ? KA.zeros(kab, Float32, W, H, CH, B) : KA.zeros(kab, Float32, 0, 0, 0, 0)
    dm_dsigma12 = train ? KA.zeros(kab, Float32, W, H, CH, B) : KA.zeros(kab, Float32, 0, 0, 0, 0)

    workgroupsize = (BLOCK_X, BLOCK_Y)
    ndrange = (cld(W, BLOCK_X) * BLOCK_X, cld(H, BLOCK_Y) * BLOCK_Y, B)
    _fused_ssim!(kab, workgroupsize)(
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12,
        img, ref, C1, C2, train; ndrange)

    return ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12
end

function fused_ssim_bwd(
    img::T, ref::T, dL_dmap::T,
    dm_dmu1::T, dm_dsigma1_sq::T, dm_dsigma12::T;
    C1::Float32 = 0.01f0^2, C2::Float32 = 0.03f0^2,
) where T <: AbstractArray{Float32, 4}
    W, H, CH, B = size(img)
    kab = get_backend(img)
    dL_dimg = KA.zeros(kab, Float32, W, H, CH, B)

    workgroupsize = (BLOCK_X, BLOCK_Y)
    ndrange = (cld(W, BLOCK_X) * BLOCK_X, cld(H, BLOCK_Y) * BLOCK_Y, B)
    _fused_ssim_bwd!(kab, workgroupsize)(
        dL_dimg, img, ref, dL_dmap,
        dm_dmu1, dm_dsigma1_sq, dm_dsigma12; ndrange)
    return dL_dimg
end

function fused_ssim(img::T; ref::T, C1::Float32 = 0.01f0^2, C2::Float32 = 0.03f0^2) where T <: AbstractArray{Float32, 4}
    train = within_gradient(img)
    y = _fused_ssim(img; ref, C1, C2, train)
    return train ? y : y[1]
end

function CRC.rrule(::typeof(_fused_ssim),
    img::T; ref::T, C1::Float32 = 0.01f0^2, C2::Float32 = 0.03f0^2, train::Bool,
) where T <: AbstractArray{Float32, 4}
    ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = _fused_ssim(img; ref, C1, C2, train)
    _pullback(Delta) = return CRC.NoTangent(), fused_ssim_bwd(
        img, ref, CRC.unthunk(Delta),
        dm_dmu1, dm_dsigma1_sq, dm_dsigma12; C1, C2)
    return ssim_map, _pullback
end
