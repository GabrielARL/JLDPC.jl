#!/usr/bin/env julia
using Statistics, Printf
using SignalAnalysis
using DSP: resample
using DataFrames
using Plots
using JLDPC.SignalUtils

# --- utils (same as before) ---
modulate(x)   = x == 1 ? -1.0 : 1.0
demodulate(x) = x < 0 ? 1 : 0
scale(x)      = x ./ maximum(abs.(x))
function mseq(degree)
    if degree == 8
        reg = ones(Int8, 8)
        seq = Vector{Float64}(undef, 2^8 - 1)
        @inbounds for i in 1:length(seq)
            seq[i] = reg[end]; feedback = xor(reg[8], reg[6]); reg = [feedback; reg[1:7]]
        end; return seq
    else; error("Only mseq(8) supported"); end
end
function add_awgn(x; EbN0_dB::Real)
    EbN0 = 10.0^(EbN0_dB/10); N0 = 1/EbN0; σ2 = N0/2
    x .+ sqrt(σ2) .* (randn(length(x)) .+ 1im .* randn(length(x)))
end
function fir_bandpass(fs, f1, f2; order::Int=128)
    @assert 0<f1<f2<fs/2
    M=order; n=0:M; m=n .- M/2
    lp(f) = @. 2*(f/fs) * sinc(2*(f/fs) * m)
    h = lp(f2) .- lp(f1); w = @. 0.54 - 0.46*cos(2π*n/M)
    h.*w
end

function main()
    # --- params (yours) ---
    fc, fs, pbfs = 24_000.0, 8_000.0, 192_000.0
    sps = Int(round(pbfs/fs))
    num_blocks = 5
    θs = range(-π, π; length=361)   # widen to full ±180° just in case
    EbN0_dB, doppler_ppm, fd_hz = 12.0, 80.0, 7.0
    Γ, jitter_std = 1e-5, π/60
    n = (2^8 - 1)*num_blocks

    # --- reference & passband ---
    bits = mseq(8)
    x_true = modulate.(repeat(bits, num_blocks))      # 1275
    base_bb = signal(x_true, fs)
    pb_sig  = scale(upconvert(base_bb, sps, fc))
    x_clean = samples(downconvert(pb_sig, sps, fc))

    # --- corrupt: ppm + CFO + AWGN ---
    ppm = doppler_ppm*1e-6
    x_skew = resample(x_clean, 1+ppm)
    t = collect(0:length(x_skew)-1) ./ fs
    x_cfo = x_skew .* exp.(1im*2π*fd_hz .* t)
    x_ch  = add_awgn(x_cfo; EbN0_dB)

    # baseline BER
    L0 = min(length(x_ch), length(x_true))
    ber_pre = mean(map(demodulate, real(x_ch[1:L0])) .!= x_true[1:L0])
    @info "Pre-recovery BER (naive) = $(round(ber_pre,digits=4))"

    # --- deskew via do_resample ---
    dopp = DataFrame(score=Float64[], idx=Int[], dop=Float64[])
    rx_rec_bb = SignalUtils.do_resample(x_ch, x_true, sps, fc, dopp, Int(pbfs), n)
    @info "do_resample() → length(rx_rec_bb) = $(length(rx_rec_bb))"

    # --- CFO fine search (tight grid around 0..±12 Hz) ---
    function best_cfo_by_mf(x, ref, fs; grid=-12.0:0.1:12.0)
        t = collect(0:length(x)-1) ./ fs
        best_fd, best_pk, best = 0.0, -Inf, x
        for fd in grid
            xr = x .* exp.(-1im*2π*fd .* t)
            pk = maximum(abs.(conv(xr, reverse(ref))))
            if pk > best_pk; best_pk = pk; best_fd = fd; best = xr; end
        end
        return best_fd, best
    end
    fd_hat, rx_rec_cfo = best_cfo_by_mf(rx_rec_bb, x_true, fs)
    @info "CFO fine search → fd_hat = $(round(fd_hat,digits=2)) Hz"

    # --- PLL + half-frequency unwrap + alignment ---
    bw = max(0.02*fc, 400.0)
    bpf = fir_bandpass(pbfs, 2*fc - bw, 2*fc + bw; order=128)
    pkt_sig = signal(rx_rec_cfo, fs)
    ref_sig = signal(x_true, fs)
    y = SignalUtils.correct_bpsk_packet_phase(pkt_sig, ref_sig, Γ, sps, fc, pbfs, n, bpf, jitter_std)

    # --- global phase snap (your estimator) ---
    y_aligned, θ_deg = SignalUtils.argminphase(y, x_true[1:length(y)])

    # --- optional polarity/angle polish (small residual) ---
    best, best_pol, best_angle = 1.0, 1.0, 0.0
    for pol in (+1.0, -1.0)
        sig = y_aligned .* pol
        for θ in range(-π/12, π/12; length=97)
            rot = sig .* cis(-θ)
            ber = mean(map(demodulate, real(rot)) .!= x_true[1:length(rot)])
            if ber < best; best=ber; best_pol=pol; best_angle=θ; end
        end
    end

    ber_post = best
    @info "Post-recovery BER = $(round(ber_post, digits=6))  (θ_snap=$(round(θ_deg, digits=2))°, pol=$(best_pol > 0 ? "+" : "-"), micro-θ=$(round(rad2deg(best_angle), digits=2))°)"


    # plots
    best_rot = y_aligned .* best_pol .* cis(-best_angle)
    scatter(real(best_rot), imag(best_rot), title="Constellation after deskew + CFO + PLL + phase snap",
            xlabel="Re", ylabel="Im", grid=true, aspect_ratio=1)
    cr0 = abs.(conv(x_ch,         reverse(x_true)))
    cr1 = abs.(conv(best_rot,     reverse(x_true)))
    p1 = plot(cr0, label="pre",  title="Matched-filter magnitude", xlabel="lag", ylabel="|corr|")
    plot!(p1, cr1, label="post"); display(p1)

    return (; ber_pre, ber_post, fd_hat, len_rec=length(rx_rec_bb), len_final=length(best_rot))
end

res = main()
@info "Summary" res
println("✅ Done.")
