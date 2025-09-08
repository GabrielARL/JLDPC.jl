#!/usr/bin/env julia

# DSP-0.7+ safe demo: AWGN + Doppler (ppm skew + CFO) → SignalUtils compensation
# Uses your params and your SignalUtils API. No firwin needed.

using Statistics, Printf
using SignalAnalysis
using DSP: resample, filtfilt
using DataFrames
using Plots
using JLDPC.SignalUtils    # if your module is top-level, switch to: using .SignalUtils

# -------------------------
# Parameters (yours)
# -------------------------
fc    = 24_000.0
fs    = 8_000.0
pbfs  = 192_000.0
sps   = Int(round(pbfs / fs))
num_blocks = 5
k     = 255                            # mseq(8) length
θs    = range(-π/8, π/8, length=100)

# Channel knobs
EbN0_dB     = 12.0                     # AWGN strength
doppler_ppm = 80.0                     # sampling skew (ppm)
fd_hz       = 7.0                      # baseband CFO (Hz)

# PLL/phase params for correct_bpsk_packet_phase
Γ           = 1e-5                     # PLL bandwidth-ish
jitter_std  = π/60                     # approximate phase jitter (rad)
n           = (2^8 - 1) * num_blocks   # number of symbols we want to recover

# -------------------------
# Utilities (matching your style)
# -------------------------
modulate(x)   = x == 1 ? -1.0 : 1.0
demodulate(x) = x < 0 ? 1 : 0
scale(x)      = x ./ maximum(abs.(x))

function mseq(degree)
    if degree == 8
        reg = ones(Int8, 8)
        seq = Vector{Float64}(undef, 2^8 - 1)
        @inbounds for i in 1:length(seq)
            seq[i] = reg[end]
            feedback = xor(reg[8], reg[6])
            reg = [feedback; reg[1:7]]
        end
        return seq
    else
        error("Only mseq(8) supported")
    end
end

# Complex AWGN at Eb/N0 for unit-energy BPSK symbols
function add_awgn(x; EbN0_dB::Real)
    EbN0 = 10.0^(EbN0_dB/10)
    N0   = 1 / EbN0
    σ2   = N0/2
    n = sqrt(σ2) .* (randn(length(x)) .+ 1im .* randn(length(x)))
    x .+ n
end

# Simple windowed-sinc FIR band-pass designer (Hamming)
# Safer than relying on DSP API variations; returns FIR taps for filtfilt(h, x)
function fir_bandpass(fs, f1, f2; order::Int=128)
    @assert f1 > 0 && f2 > f1 && f2 < fs/2 "Band edges must be within (0, fs/2)"
    M = order
    n = 0:M
    m = n .- M/2
    lp(f) = @. 2*(f/fs) * sinc(2*(f/fs) * m)    # ideal lowpass to f (Hz)
    h = lp(f2) .- lp(f1)                         # ideal bandpass = lp(f2) - lp(f1)
    w = @. 0.54 - 0.46 * cos(2π * n / M)         # Hamming window
    h .* w
end

# Matched-filter peak helper
mf_peak(x, tpl) = maximum(abs.(conv(x, reverse(tpl))))

# -------------------------
# 1) Build reference baseband & passband
# -------------------------
bits    = mseq(8)
x_true  = modulate.(repeat(bits, num_blocks))          # length = 1275
base_bb = signal(x_true, fs)                           # complex baseband wrapper

pb_sig  = scale(upconvert(base_bb, sps, fc))           # real passband @ pbfs
rx_bb_clean = downconvert(pb_sig, sps, fc)             # ideal back to baseband
x_clean = samples(rx_bb_clean)                         # ComplexF64 vector @ fs

# -------------------------
# 2) Corrupt with Doppler (ppm skew + CFO) and AWGN
# -------------------------
ppm  = doppler_ppm * 1e-6
x_skew = resample(x_clean, 1 + ppm)                    # clock skew (simulate SRO)
t      = collect(0:length(x_skew)-1) ./ fs
x_cfo  = x_skew .* exp.(1im * 2π * fd_hz .* t)         # CFO in baseband
x_ch   = add_awgn(x_cfo; EbN0_dB=EbN0_dB)              # noisy, skewed, rotated

# Baseline BER without any recovery
L0      = min(length(x_ch), length(x_true))
ber_pre = mean(map(demodulate, real(x_ch[1:L0])) .!= x_true[1:L0])
@info "Pre-recovery BER (naive) = $(round(ber_pre, digits=4))"

# -------------------------
# 3) Doppler compensation via your SignalUtils.do_resample
#    do_resample(rx_pkt, x_ref, sps, fc, dopp::DataFrame, pbfs::Int, n::Int)
#    - rx_pkt, x_ref are baseband vectors at fs = pbfs/sps (=> fs here)
# -------------------------
dopp = DataFrame(score=Float64[], idx=Int[], dop=Float64[])
rx_rec_bb = SignalUtils.do_resample(x_ch, x_true, sps, fc, dopp, Int(pbfs), n)

# -------------------------
# 4) Phase/PLL cleanup via your SignalUtils.correct_bpsk_packet_phase
#    Needs a band-pass FIR around 2*fc applied on squared-signal path.
#    We'll design taps ourselves (windowed-sinc), then pass taps to your function.
# -------------------------
bw = max(0.02 * fc, 400.0)     # about 2% of fc (min 400 Hz)
bpf = fir_bandpass(pbfs, 2*fc - bw, 2*fc + bw; order=128)

pkt_sig = signal(rx_rec_bb, fs)                # SampledSignal for API
ref_sig = signal(x_true,   fs)

rx_final = SignalUtils.correct_bpsk_packet_phase(
    pkt_sig, ref_sig, Γ, sps, fc, pbfs, n, bpf, jitter_std
)

# -------------------------
# 5) Optional final polish (polarity/angle sweep) + BER
# -------------------------
best = 1.0
best_pol, best_angle = 1.0, 0.0
for pol in (+1.0, -1.0)
    sig = rx_final .* pol
    for θ in θs
        rotated = sig .* cis(-θ)
        ber = mean(map(demodulate, real(rotated)) .!= x_true[1:length(rotated)])
        if ber < best
            best = ber
            best_pol = pol
            best_angle = θ
        end
    end
end

ber_post = best
@info "Post-recovery BER = $(round(ber_post, digits=6)) (pol=$(best_pol>0 ? "+" : "-"), angle=$(round(rad2deg(best_angle),digits=2))°)"

# -------------------------
# 6) Constellation + MF plots
# -------------------------
best_rot = rx_final .* best_pol .* cis(-best_angle)
scatter(real(best_rot), imag(best_rot),
    title="Constellation after do_resample + packet PLL",
    xlabel="Re", ylabel="Im", grid=true, aspect_ratio=1)

cr0 = abs.(conv(x_ch,     reverse(x_true)))
cr1 = abs.(conv(rx_final, reverse(x_true)))
p1 = plot(cr0, label="pre",  title="Matched-filter magnitude", xlabel="lag", ylabel="|corr|")
plot!(p1, cr1, label="post")
display(p1)

println("✅ Done.")
