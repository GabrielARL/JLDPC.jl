module CFO_est
# Pilot-aided CFO + channel/equalizer helpers

export bits_pm1, rrc, chpb, up, down, align, awgn,
       cfo1, derot, dfe, berp, pb_impair, dd_pll

using Random, Statistics
using SignalAnalysis          # signal, upconvert, downconvert, delay, samples
using AdaptiveEstimators      # DFE, RLS, fit!, nearest
using DSP: unwrap, hilbert    # hilbert for passband impairments

# ---------------- bits & pulses ----------------
"""
    bits_pm1(N) -> Vector{Float64}
Random ±1 bits as Float64.
"""
bits_pm1(N) = ifelse.(rand(Bool, N), 1.0, -1.0)

"""
    rrc(β, sps, span) -> taps
Square-root raised-cosine taps; length = span*sps + 1.
"""
rrc(β, sps, span) = rrcosfir(β, sps, span)

# ---------------- passband channel ----------------
"""
    chpb(paths, pbfs) -> h::Vector{Float64}
Build real-FIR passband channel from `paths = [(τ_sec, gain_dB), ...]`, energy-normalized.
"""
function chpb(paths::AbstractVector{<:Tuple{<:Real, <:Real}}, pbfs::Real)
    dsmpl = round.(Int, first.(paths) .* pbfs)
    glin  = 10 .^ (last.(paths) ./ 20)
    h = zeros(Float64, maximum(dsmpl) + 1)
    @inbounds for (d, g) in zip(dsmpl, glin)
        h[d + 1] += g
    end
    h ./= sqrt(sum(abs2, h) + eps())
    return h
end

# ---------------- tx / rx ----------------
"""
    up(xb, fs, sps, fc, taps) -> y_pb::Vector{Float64}
BPSK baseband @ fs → passband real @ pbfs = fs*sps (interp + SRRC + mix).
"""
function up(xb::AbstractVector{<:Real}, fs::Real, sps::Int, fc::Real, taps)
    x = signal(complex.(xb), fs)
    samples(upconvert(x, sps, fc, taps))
end

"""
    down(y_pb, pbfs, sps, fc, taps) -> (z_full, symdelay, pfrac)
Passband real → symbol-rate complex (mix + SRRC + decimate), with SRRC delay info.
"""
function down(y_pb::AbstractVector{<:Real}, pbfs::Real, sps::Int, fc::Real, taps)
    L      = length(taps)
    symdel = (L - 1) ÷ sps
    pfrac  = ((L - 1) ÷ 2) % sps
    rx_del = delay(signal(y_pb, pbfs), pfrac)
    bb     = downconvert(rx_del, sps, fc, taps)
    return samples(bb), symdel, pfrac
end

# ---------------- coarse timing ----------------
"""
    align(z_full, xb, Np, symdel; sweep=-2:2) -> (z, soff)
Pick the best integer-symbol offset by maximizing pilot coherence.
"""
function align(z_full::AbstractVector{<:Complex}, xb::AbstractVector{<:Real},
               Np::Int, symdel::Int; sweep = -2:2)
    k = length(xb)
    best, bsoff = -Inf, 0
    bestslice = view(z_full, 1:1)
    for s in sweep
        s0, s1 = symdel + s + 1, symdel + s + k
        (s0 < 1 || s1 > length(z_full)) && continue
        z = @view z_full[s0:s1]
        sc = abs(sum(@view(z[1:Np]) .* complex.(@view(xb[1:Np]))))
        if sc > best
            best, bsoff, bestslice = sc, s, z
        end
    end
    return copy(bestslice), bsoff
end

# ---------------- symbol-rate AWGN ----------------
"""
    awgn(z, EbN0_dB) -> z_noisy
Complex AWGN at symbol rate, calibrated by post-MF Es.
"""
function awgn(z::AbstractVector{<:Complex}, ebn0_db::Real)
    Es = mean(abs2, z)
    N0 = Es / (10.0^(ebn0_db/10))
    σ  = sqrt(N0/2)
    z .+ σ .* (randn(length(z)) .+ 1im .* randn(length(z)))
end

# ---------------- passband impairments ----------------
"""
    pb_impair(y_pb, fs_pb; fD_hz=0.0, pn_std=0.0) -> y_pb_imp
Apply passband Doppler (Hz) + Wiener phase noise (rad increment std per PB sample)
by rotating the analytic passband signal; return real waveform.
"""
function pb_impair(y_pb::AbstractVector{<:Real}, fs_pb::Real; fD_hz::Real=0.0, pn_std::Real=0.0)
    N  = length(y_pb)
    xa = hilbert(y_pb)                        # analytic (complex)
    n  = 0:N-1
    θd = 2π .* fD_hz .* (n ./ fs_pb)          # Doppler phase
    Δφ = pn_std .* randn(N)                   # PN increments
    φ  = cumsum(Δφ)                           # Wiener PN
    real(xa .* cis.(θd .+ φ))                 # back to real PB
end

# ---------------- pilot-aided CFO (1 sps, BPSK) ----------------
"""
    cfo1(zp, xp, fs) -> (fhat_Hz, dphi_rad_per_sym, ok)
Demod pilot (±1), remove bulk phase, lag-1 Kay; apply only if pilot coherence improves.
"""
function cfo1(zp::AbstractVector{<:Complex}, xp::AbstractVector{<:Real}, fs::Real)
    @assert length(zp) == length(xp)
    y  = zp .* complex.(xp)                 # remove ±1
    φb = angle(sum(y))                      # bulk phase
    y0 = y .* cis.(-φb)

    num  = sum(y0[2:end] .* conj.(y0[1:end-1]))  # lag-1 Kay
    dphi = angle(num)                              # rad/sample
    fhat = (fs/(2π)) * dphi                        # Hz

    # guard: only apply if pilot coherence improves
    cb = abs(sum(zp .* complex.(xp)))
    n  = 0:length(zp)-1
    zt = zp .* cis.(-dphi .* n)
    ca = abs(sum(zt .* complex.(xp)))
    ok = ca >= cb

    return fhat, dphi, ok
end

"""
    derot(z, dphi) -> z_rot
Apply per-sample phase correction e^{-j dphi n}.
"""
derot(z::AbstractVector{<:Complex}, dphi::Real) = begin
    n = 0:length(z)-1
    z .* cis.(-dphi .* n)
end

# ---------------- decision-directed PLL (symbol rate) ----------------
"""
    dd_pll(y; Q=[1,-1], μ=0.02, α=0.002, phi0=0.0) -> (y_corr, phi)
Decision-directed PI carrier loop at symbol rate (good post-DFE).
- Q   : constellation (Complex), e.g. ComplexF64[1,-1] for BPSK
- μ   : proportional gain
- α   : integrator gain
- phi0: initial phase (rad)
"""
function dd_pll(y::AbstractVector{<:Complex};
                Q = ComplexF64[1.0, -1.0], μ::Real=0.02, α::Real=0.002, phi0::Real=0.0)
    N    = length(y)
    yc   = similar(y)
    phi  = zeros(Float64, N)
    dec  = nearest(Q)
    ph   = phi0
    acc  = 0.0
    @inbounds for n in 1:N
        z = y[n] * cis(-ph)          # derotate by current estimate
        d = dec(z)                   # nearest symbol
        e = imag(z * conj(d))        # DD phase error (Costas-like)
        acc += α * e                 # integrator
        ph  += μ * e + acc           # PI update
        yc[n]  = z
        phi[n] = ph
    end
    return yc, phi
end

# ---------------- equalization & BER ----------------
"""
    dfe(z, train, total_len; ff=40, fb=40) -> y_eq
Train RLS-DFE with known `train` (pilot), return equalized stream.
"""
function dfe(z::AbstractVector{<:Complex}, train::AbstractVector{<:Complex}, total_len::Int;
             ff::Int=40, fb::Int=40)
    dec = nearest([1.0 + 0im, -1.0 + 0im])
    r   = fit!(DFE(ComplexF64, ff, fb), RLS(), z, train, total_len, dec)
    ϕ   = angle(sum(r.y[1:length(train)] .* conj.(train)))   # tiny bulk phase
    r.y .* cis(-ϕ)
end

"""
    berp(y, xb, Np) -> ber
Payload-only BER with BPSK hard decision.
"""
function berp(y::AbstractVector{<:Complex}, xb::AbstractVector{<:Real}, Np::Int)
    dec = nearest([1.0 + 0im, -1.0 + 0im])
    ypay = @view y[Np+1:end]
    xref = complex.(@view xb[Np+1:end])
    h    = dec.(ypay)
    count(h .!= xref) / length(xref)
end

end # module CFO_est
