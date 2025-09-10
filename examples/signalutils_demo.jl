using JLDPC
using JLDPC.SignalUtils
using SignalAnalysis
using Random
# using Plots  # optional

Random.seed!(123)

# --- 1) Synthetic BPSK training sequence ---
N = 256
x_bits = rand(0:1, N)
x_bpsk = ComplexF64.(ifelse.(x_bits .== 1, 1.0, -1.0))

# --- 2) Sparse channel + noise ---
L_h = 12
true_h = zeros(ComplexF64, L_h)
true_h[[3, 7]] .= [0.85 + 0.12im, 0.55 - 0.22im]        # 2-tap sparse
y_clean = myconv(x_bpsk, true_h)
σ = 0.08
y = y_clean .+ σ .* (randn(ComplexF64, length(y_clean)) .+ 1im .* randn(length(y_clean))) ./ √2

# --- 3) Channel estimation (MMSE + sparse ISTA) ---
h_mmse   = estimate_mmse_channel(y[1:N], x_bpsk[1:N], L_h; σ²=σ^2)
h_sparse = estimate_sparse_channel(y[1:N], x_bpsk[1:N], L_h; σ²=σ^2, λ=1e-2)

println("True h:     ", true_h)
println("MMSE est:   ", round.(h_mmse, digits=3))
println("Sparse est: ", round.(h_sparse, digits=3))

# --- 4) Matched filter (correlation) to reveal timing peaks ---
# Use SampledSignal to leverage SignalAnalysis’ mfilter nicely
fs = 1.0
x_sig = signal(x_bpsk, fs)
y_sig = signal(y, fs)

cr = abs.(mfilter(x_sig, y_sig))     # correlation magnitude -> sharp peaks
peaks = allpeaks(cr)                  # now there should be peaks
p1 = firstpeak(cr)

println("Detected correlation peaks: ", peaks[1:min(end, 10)])  # show first few
println("First strong peak index: ", p1)

