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

end # module
