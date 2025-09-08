module SignalUtils

using Statistics
using LinearAlgebra
using SparseArrays
using DataFrames
using SignalAnalysis
using DSP: filtfilt

const SampledSignal = SignalAnalysis.SampledSignal

export allpeaks, firstpeak, stdize,
       estimate_mmse_channel, estimate_sparse_channel,
       estimate_sparse_ls, estimate_omp_channel,
       do_resample, track_bpsk_carrier_pll,
       unwrap_half_frequency, correct_bpsk_phase_shift,
       correct_bpsk_packet_phase, argminphase, argminphase_ber

# -----------------------
# Small helpers
# -----------------------
_as_samples(y) = y isa SampledSignal ? samples(y) : y
_as_signal(x, fs) = x isa SampledSignal ? x : signal(x, fs)

# -----------------------
# Peak Detectors
# -----------------------
function allpeaks(x::AbstractVector{<:Real})
    θ = max(maximum(x) / 2.0, 3.5 * median(x))
    peaks = Int[]
    start_index = 1
    while start_index <= length(x)
        next_peak_pos = findfirst(@view(x[start_index:end]) .>= θ)
        next_peak_pos === nothing && break
        p = start_index + next_peak_pos - 1
        while p < length(x) && x[p+1] > x[p]
            p += 1
        end
        push!(peaks, p)
        start_index = p + 1
    end
    return peaks
end

function firstpeak(x::AbstractVector{<:Real})
    θ = max(maximum(x) / 1.5, 3.5 * median(x))
    p = findfirst(x .≥ θ)
    while p !== nothing && p < length(x) && x[p+1] > x[p]
        p += 1
    end
    return p
end

function stdize(y)
    v = _as_samples(y)
    ȳ = mean(v)
    μ = v .- ȳ
    σ = sqrt(mean(abs2, μ)) + eps()
    return μ ./ σ
end

# -----------------------
# Channel Estimation
# -----------------------
function estimate_mmse_channel(y_train::AbstractVector, x_train::AbstractVector, L_h::Int; σ²::Float64=1e-3)
    N = length(y_train)
    @assert length(x_train) ≥ N "x_train must be at least as long as y_train"
    X = zeros(ComplexF64, N, L_h)
    for i in 1:N, j in 1:L_h
        if i - j + 1 ≥ 1
            X[i, j] = x_train[i - j + 1]
        end
    end
    XtX = X' * X
    return (XtX + σ² * I(L_h)) \ (X' * y_train)
end

function estimate_sparse_channel(y_train::AbstractVector, x_train::AbstractVector, L_h::Int; σ²::Float64=1e-3, λ::Float64=1e-2, iters::Int=120)
    N = length(y_train)
    @assert length(x_train) ≥ N "x_train must be at least as long as y_train"
    X = zeros(ComplexF64, N, L_h)
    for i in 1:N, j in 1:L_h
        if i - j + 1 ≥ 1
            X[i, j] = x_train[i - j + 1]
        end
    end
    h = zeros(ComplexF64, L_h)
    α = 1.0 / (opnorm(X)^2 + σ²)
    @inbounds for _ in 1:iters
        grad = X' * (X * h - y_train) + σ² * h
        h_temp = h - α * grad
        # complex soft threshold (apply to magnitude)
        mag = abs.(h_temp)
        scale = max.(mag .- α .* λ, 0.0) ./ (mag .+ eps())
        h = h_temp .* scale
    end
    return h
end

function estimate_sparse_ls(y_train::AbstractVector, x_train::AbstractVector, L_h::Int, k::Int)
    N = length(y_train)
    X = zeros(ComplexF64, N, L_h)
    for i in 1:N, j in 1:L_h
        if i - j + 1 ≥ 1
            X[i, j] = x_train[i - j + 1]
        end
    end
    h_ls = X \ y_train
    idx = partialsortperm(abs.(h_ls), rev=true, 1:k)
    h_sparse = zeros(ComplexF64, L_h)
    h_sparse[idx] = h_ls[idx]
    return h_sparse
end

function estimate_omp_channel(y_train::AbstractVector, x_train::AbstractVector, L_h::Int, k::Int)
    N = length(y_train)
    X = zeros(ComplexF64, N, L_h)
    for i in 1:N, j in 1:L_h
        if i - j + 1 ≥ 1
            X[i, j] = x_train[i - j + 1]
        end
    end
    residual = copy(y_train)
    support = Int[]
    @inbounds for _ in 1:k
        correlations = abs.(X' * residual)
        j = argmax(correlations)
        push!(support, j)
        Xs = X[:, support]
        h_tmp = Xs \ y_train
        residual = y_train - Xs * h_tmp
    end
    h_omp = zeros(ComplexF64, L_h)
    h_omp[support] = X[:, support] \ y_train
    return h_omp
end

# -----------------------
# Resample and PLL
# -----------------------
function do_resample(rx_pkt::AbstractVector, x_ref::AbstractVector, sps::Int, fc::Real,
                     dopp::DataFrame, pbfs::Int, n::Int)

    # 1) Upconvert both rx & ref to passband
    rx_pb  = upconvert(signal(rx_pkt, pbfs / sps), sps, fc)
    ref_pb = upconvert(signal(x_ref, pbfs / sps), sps, fc)

    # 2) Scan Doppler via resample factor
    best_score = -Inf
    best_dop   = 0.0
    best_idx   = 0

    for dop in -0.98:0.02:1.0
        factor = 1 / (1 + dop)
        isfinite(factor) || continue

        resampled = signal(resample(rx_pb, factor), pbfs)
        cr = mfilter(ref_pb, resampled)
        val, idx = findmax(abs.(cr))

        push!(dopp, (real(val), idx, real(dop)))
        if val > best_score
            best_score, best_dop, best_idx = val, dop, idx
        end
    end

    # 3) Apply best factor → downconvert to baseband
    opt_factor = 1 / (1 + best_dop)
    rx_pb_corr = signal(resample(rx_pb, opt_factor), pbfs)
    rx_bb      = downconvert(rx_pb_corr, sps, fc)

    # 4) Align to reference in baseband
    cr = mfilter(signal(x_ref, pbfs / sps), rx_bb)
    _, align_idx = findmax(abs.(cr))

    # 5) Bounds-safe extraction of n samples around align_idx
    v = _as_samples(rx_bb)   # index raw samples, not the MetaArray wrapper
    L = length(v)

    # If requested n is longer than we have, truncate
    n_take = min(n, L)

    # Center the window on the correlation peak when possible
    start = clamp(align_idx - (n_take - 1) ÷ 2, 1, max(1, L - n_take + 1))
    stop  = start + n_take - 1

    return v[start:stop]
end

function track_bpsk_carrier_pll(x, fc, fs, bandwidth=1e-5)
    β = √bandwidth
    ϕ = 0.0
    ω = 0.0
    y = zeros(ComplexF64, length(x))
    demodulated = zeros(length(x))
    phase_errors = zeros(length(x))
    @inbounds for j in 1:length(x)
        y[j] = cis(-2π * fc * (j-1)/fs + ϕ)
        phase_error = angle(x[j] * conj(y[j]))
        ω += bandwidth * phase_error
        ϕ += β * phase_error + ω
        demodulated[j] = real(x[j] * conj(y[j]))
        phase_errors[j] = phase_error
    end
    return (signal(y, fs), signal(demodulated, fs), signal(phase_errors, fs))
end

function unwrap_half_frequency(sig2fc)
    ph_2f = angle.(_as_samples(sig2fc))
    ph_2f_unwrap = unwrap(ph_2f)
    ph_fc = ph_2f_unwrap ./ 2
    return cis.(ph_fc)
end

# -----------------------
# Phase Correction
# -----------------------
function correct_bpsk_phase_shift(packets, x_datas, Γ, i, spsd, fc, pbfs, n, bpf)
    y_data = stdize(packets[i, 1:n])
    yp_pb = upconvert(signal(y_data, pbfs / spsd), spsd, fc)
    y1, _, _ = track_bpsk_carrier_pll(yp_pb, fc, pbfs, Γ)
    ysq_filt = filtfilt(bpf, _as_samples(yp_pb).^2)
    y = pll(signal(ysq_filt, framerate(yp_pb)), 2*fc, 1e-5; fs=pbfs)
    y_half = unwrap_half_frequency(y)
    phase_error = angle.(_as_samples(y1) .* conj(y_half))
    ϕ = unwrap(phase_error)
    y_pb_pll = signal(_as_samples(yp_pb) .* exp.(-im .* ϕ), framerate(yp_pb))
    y_bb_pll = downconvert(y_pb_pll, spsd, fc)
    ref_data = x_datas[i]
    cr = mfilter(signal(ref_data, framerate(y_bb_pll)), y_bb_pll)
    _, ixd = findmax(abs.(cr))

    # ---- bounds-safe extraction (centered + clamped) ----
    v = _as_samples(y_bb_pll)
    L = length(v)
    n_take = min(n, L)
    start = clamp(ixd - (n_take - 1) ÷ 2, 1, max(1, L - n_take + 1))
    stop  = start + n_take - 1

    return v[start:stop]
end

function correct_bpsk_packet_phase(pkt::SampledSignal, x_ref::SampledSignal, Γ::Float64, sps::Int,
                                   fc::Float64, pbfs::Float64, n::Int, bpf::AbstractVector,
                                   jitter_std::Float64)
    y_data = stdize(pkt)
    yp_pb = upconvert(signal(y_data, framerate(pkt)), sps, fc)
    y1, _, _ = track_bpsk_carrier_pll(yp_pb, fc, pbfs, Γ)
    ysq_filt = filtfilt(bpf, _as_samples(yp_pb).^2)
    ysq_sig  = signal(ysq_filt, framerate(yp_pb))
    pll_bandwidth = jitter_std ≤ π/48 ? 5e-6 : jitter_std ≤ π/90 ? 1e-6 : 1e-5
    y = pll(ysq_sig, 2*fc, pll_bandwidth; fs=pbfs)
    y_half = unwrap_half_frequency(y)
    phase_error = angle.(_as_samples(y1) .* conj(y_half))
    ϕ = unwrap(phase_error)
    corrected_samples = _as_samples(yp_pb) .* exp.(-im .* ϕ)
    y_pb_pll = signal(corrected_samples, framerate(yp_pb))
    y_bb_pll = downconvert(y_pb_pll, sps, fc)
    cr = mfilter(x_ref, y_bb_pll)
    _, ixd = findmax(abs.(cr))

    # ---- bounds-safe extraction (centered + clamped) ----
    v = _as_samples(y_bb_pll)
    L = length(v)
    n_take = min(n, L)
    start = clamp(ixd - (n_take - 1) ÷ 2, 1, max(1, L - n_take + 1))
    stop  = start + n_take - 1

    return v[start:stop]
end

# -----------------------
# Phase Alignment Estimators
# -----------------------
# BER-based (original): robust for big phase, but flat near 0° for BPSK
function argminphase_ber(x̂, x)
    bers = Float64[]
    range = 0.0:0.1:360.0
    for θ in range
        x̂a = x̂ .* exp(-im * deg2rad(-θ))
        ber = sum(abs, sign.(real.(x̂a)) .!= sign.(real.(x)))
        push!(bers, ber)
    end
    val, id = findmin(bers)
    return (x̂ .* exp(-im * deg2rad(-range[id])), range[id])
end

# Correlation-phase (recommended): closed-form small-angle estimator
function argminphase(x̂, x)
    θ = -angle(sum(x̂ .* conj(x)))    # align x̂ to x
    x̂_aligned = x̂ .* exp(im*θ)
    return (x̂_aligned, rad2deg(-θ))  # report estimated original rotation in degrees
end

end # module
