# examples/signalutils_scenarios.jl
# Showcase scenarios specifically designed for JLDPC.SignalUtils utilities.
# - Scenario 1: Doppler search + resample + coarse timing via do_resample
# - Scenario 2: Carrier PLL + phase jitter cleanup via correct_bpsk_packet_phase
# - Scenario 3: Sparse multipath channel estimation (MMSE / ISTA / OMP)

using JLDPC
using JLDPC.SignalUtils
using SignalAnalysis
using DSP
using Random
using Statistics
using Printf
using DataFrames
using StatsBase
const SS = SignalAnalysis

# -----------------------
# Helpers
# -----------------------
modulate_bit(b)   = b == 1 ? (1.0 + 0im) : (-1.0 + 0im)
demod_bpsk(z)     = real(z) < 0 ? 0 : 1
mseq8() = begin
    reg = ones(Int8, 8); seq = Vector{Float64}(undef, 2^8 - 1)
    @inbounds for i in 1:length(seq)
        seq[i] = reg[end]
        fb = xor(reg[8], reg[6])
        reg = (Int8[fb]; reg[1:7])
    end
    seq
end
ber(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer}) =
    sum(a .!= b) / length(a)

# build training/reference vector (BPSK)
function make_ref(num_blocks::Int)
    bits = mseq8()
    xbits = Int.(repeat(bits, num_blocks))
    xb    = modulate_bit.(xbits)
    return xbits, xb
end

# apply sparse channel (complex taps at indices idx) and AWGN
function apply_channel(x::AbstractVector{<:Number}, taps::Vector{ComplexF64}, idx::Vector{Int}; σ=0.05)
    L = maximum(idx)
    h = zeros(ComplexF64, L)
    h[idx] .= taps
    y = myconv(x, h)
    y .+ (σ/√2) .* (randn(length(y)) .+ 1im .* randn(length(y)))
end

# simple Doppler model: resample factor 1/(1+dopp)
function apply_doppler(sig::SS.SampledSignal, dopp::Float64)
    fac = 1 / (1 + dopp)
    SS.signal(SS.resample(sig, fac), SS.framerate(sig))
end

# -----------------------
# Scenario 1 — Doppler & coarse timing using do_resample
# -----------------------
function scenario1(; fc=12_000.0, fs=8_000.0, pbfs=192_000.0, num_blocks=4, n_take=1500)
    Random.seed!(42)
    sps = Int(round(pbfs / fs))

    # reference
    xbits, xb = make_ref(num_blocks)
    x_ref = SS.signal(xb, fs)

    # build passband Tx
    tx_pb = SS.upconvert(SS.signal(xb, fs), sps, fc)
    tx_pb = SS.signal(samples(tx_pb) ./ (maximum(abs.(samples(tx_pb))) + eps()), SS.framerate(tx_pb))

    # channel: sparse 3-tap + phase + noise
    taps = [0.9 + 0.0im, 0.5 - 0.2im, 0.25 + 0.1im]; idx = [2, 6, 13]
    y_ch = apply_channel(complex.(samples(tx_pb)), taps, idx; σ=0.02)
    rx_pb = SS.signal(y_ch, SS.framerate(tx_pb))

    # inject Doppler (e.g., +0.4%) and crop a window
    dop_true = 0.004
    rx_pb_d = apply_doppler(rx_pb, dop_true)
    rx_pb_d = SS.signal(samples(rx_pb_d)[1:n_take], SS.framerate(rx_pb_d))

    # run coarse Doppler/timing search via do_resample
    df = DataFrame(score=Float64[], idx=Int[], dop=Float64[])
    rx_bb_aligned = do_resample(samples(rx_pb_d), samples(x_ref), sps, fc, df, Int(pbfs), length(xb))

    # detect peaks on correlation to validate timing
    cr = abs.(mfilter(SS.signal(xb, fs), SS.signal(rx_bb_aligned, fs)))
    peaks = allpeaks(cr)
    p1 = firstpeak(cr)

    best_row = argmax(df.score)
@info "Scenario 1: Doppler + timing" dop_true dop_best=df[best_row, :dop] nrows=size(df,1)
    println(@sprintf("  First peak at lag: %d", p1 === nothing ? -1 : p1))
    println("  Top-5 Doppler candidates (score, idx, dop):")
    for r in eachrow(first(sort(df, :score, rev=true), min(5, nrow(df))))
        @printf("    %.2f  %6d  %+0.4f\n", r.score, r.idx, r.dop)
    end
end

# -----------------------
# Scenario 2 — Carrier jitter, use PLL & packet phase correction
# -----------------------
function scenario2(; fc=18_000.0, fs=8_000.0, pbfs=192_000.0, num_blocks=3, jitter_std=π/40)
    Random.seed!(7)
    sps = Int(round(pbfs / fs))

    # build reference baseband and passband with injected phase jitter
    xbits, xb = make_ref(num_blocks)
    base = SS.signal(xb, fs)
    pb   = SS.upconvert(base, sps, fc)

    # jitter as a cumulative random walk in phase at passband
    ϕ = cumsum(randn(length(pb)) .* (jitter_std/20))
    y_pb = SS.signal(samples(pb) .* exp.(1im .* ϕ), SS.framerate(pb))

    # bandpass around 2*fc for PLL helper (squared signal has tone at 2fc)
    bpf = fir(255, 2*fc - 600, 2*fc + 600; fs=pbfs)

    # correct packet phase (PLL-based) and align to reference
    n = length(xb)
    y_corr = correct_bpsk_packet_phase(y_pb, base, 5e-6, sps, fc, pbfs, n, bpf, jitter_std)

    # quick BER vs truth
    bits_rx = demod_bpsk.(y_corr)
    println(@sprintf("Scenario 2: BER after PLL correction = %.4f", ber(bits_rx, xbits)))
end

# -----------------------
# Scenario 3 — Sparse channel estimation (MMSE / ISTA / OMP / LS-k)
# -----------------------
function scenario3(; fs=8_000.0, num_blocks=5, L_h=40, σ=0.05)
    Random.seed!(99)
    xbits, xb = make_ref(num_blocks)

    # generate sparse channel
    k_true = 5
    idx = sort(sample(2:L_h, k_true; replace=false))
    taps = randn(k_true) .+ 0.3im .* randn(k_true)
    taps ./= norm(taps)

    y = apply_channel(xb, taps, idx; σ=σ)

    # take a training-length equal to length(xb)
    y_tr = y[1:length(xb)]
    x_tr = xb[1:length(xb)]

    # estimates
    h_mmse   = estimate_mmse_channel(y_tr, x_tr, L_h; σ²=σ^2)
    h_ista   = estimate_sparse_channel(y_tr, x_tr, L_h; σ²=σ^2, λ=5e-3)
    h_ls_k   = estimate_sparse_ls(y_tr, x_tr, L_h, k_true)
    h_omp    = estimate_omp_channel(y_tr, x_tr, L_h, k_true)

    # support & MSE
    function support(v; thr=0.1)
        I = findall(abs.(v) .>= thr*maximum(abs.(v)))
        sort!(I)
        return I
    end
    sup_true = idx
    sup_mmse = support(h_mmse)
    sup_ista = support(h_ista)
    sup_ls_k = support(h_ls_k)
    sup_omp  = support(h_omp)

    mse(v) = mean(abs2, v .- let h=zeros(ComplexF64, L_h); h[idx].=taps; h end)

    println("Scenario 3: Sparse channel estimation summary")
    @printf("  True support:    %s\n", string(sup_true))
    @printf("  MMSE support:    %s  (MSE=%.4g)\n", string(sup_mmse), mse(h_mmse))
    @printf("  ISTA support:    %s  (MSE=%.4g)\n", string(sup_ista), mse(h_ista))
    @printf("  LS-k support:    %s  (MSE=%.4g)\n", string(sup_ls_k), mse(h_ls_k))
    @printf("  OMP support:     %s  (MSE=%.4g)\n", string(sup_omp),  mse(h_omp))
end

# -----------------------
# Main
# -----------------------
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n=== Running SignalUtils Scenarios ===")
    scenario1()
    println("\n-------------------------------------")
    scenario2()
    println("\n-------------------------------------")
    scenario3()
end
