# examples/bpsk_pathA_multipath_dfe_pn_doppler_passband_sweep_pll.jl
using Random, Statistics
using JLDPC.CFO_est
using DSP: conv

Random.seed!(0xC0FFEE)

# ---------- System params ----------
fs         = 8_000.0
pbfs       = 192_000.0
sps        = Int(pbfs ÷ fs); @assert sps*fs == pbfs
fc         = 24_000.0
β          = 0.25
span       = 6
ffsz, fbsz = 16, 16
k_train    = 128

# Passband impairments
sigma_pn_pb = 1.5e-3
fD_hz       = 3.0

# Sweep controls
snr_grid_db = 0.0:1.0:10.0
nframes     = 50
kbits       = 2048

# ---------- Build pulse once ----------
taps = rrc(β, sps, span)

# ---------- Channel (passband FIR) ----------
paths = [(0.0, 0.0), (0.7e-3, -3.0), (1.6e-3, -6.0)]
h_pb  = chpb(paths, pbfs)

# ---------- One-frame (no AWGN yet) ----------
function tx_through_channel_noawgn(x_bits::Vector{Float64}, taps, fc, fs, sps, pbfs)
    tx_pb = up(x_bits, fs, sps, fc, taps)        # real passband
    y_pb  = conv(tx_pb, h_pb)                    # multipath PB FIR
    y_pb  = pb_impair(y_pb, pbfs; fD_hz=fD_hz, pn_std=sigma_pn_pb)

    z_full, symdelay, _ = down(y_pb, pbfs, sps, fc, taps)

    # align by pilot coherence (use first k_train as pilot)
    Np = min(k_train, length(x_bits))
    z_sym, soff = align(z_full, x_bits, Np, symdelay; sweep=-2:2)
    return (; z_sym, symdelay, soff)
end

# ---------- Per-frame noisy run & BER ----------
function run_one_frame(ebno_db::Float64, taps)
    xb   = bits_pm1(kbits)
    chan = tx_through_channel_noawgn(xb, taps, fc, fs, sps, pbfs)
    zc   = chan.z_sym

    zn   = awgn(zc, ebno_db)

    Np    = min(k_train, length(xb))
    train = complex.(xb[1:Np])
    yeq   = dfe(zn, train, length(xb); ff=ffsz, fb=fbsz)

    phi0  = angle(sum(@view(yeq[1:Np]) .* conj.(train)))
    ypll, _phi = dd_pll(yeq; μ=0.02, α=0.002, phi0=phi0)

    berp(ypll, xb, Np)   # payload BER
end

# ---------- Sweep ----------
println("SNR sweep 0 → 10 dB (true Eb/N0 @ symbol rate) with DFE + DD-PLL")
println("nframes=$(nframes), kbits=$(kbits), PN_pb=$(sigma_pn_pb), fD=$(fD_hz) Hz")
for ebno_db in snr_grid_db
    bers = Float64[]
    for _ in 1:nframes
        push!(bers, run_one_frame(ebno_db, taps))
    end
    println("Eb/N0 = $(ebno_db) dB → BER = ", mean(bers), "  (±", std(bers), ")")
end
