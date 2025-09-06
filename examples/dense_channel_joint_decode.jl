#!/usr/bin/env julia
using Random, LinearAlgebra
using JLDPC

# ----------------------------
# Config
# ----------------------------
Random.seed!(123)
k, n, npc = 64, 128, 4
Lh        = 9                # dense channel length
snr_db    = 12.0             # SNR in dB

# ----------------------------
# Load LDPC code + structure
# ----------------------------
code, cols, idrows, pilot_idx = JLDPC.initcode(k, n, npc)
Hs = JLDPC.get_H_sparse(code)
println("Loaded code: $(code)  (H size = ", size(code.H), ")")

# sanity: all-zero codeword check over H
x0 = zeros(Int, n)
println("Parity check OK? ", JLDPC.is_valid_codeword(x0, Hs))

# ----------------------------
# Build an all-zero codeword & BPSK transmit
# ----------------------------
bits  = zeros(Int, k)              # all-zero message
x_enc = JLDPC.encode(code, bits)   # length n, in {0,1}

# Correct elementwise mapping: 0->-1, 1->+1
s = ifelse.(x_enc .== 1, 1.0, -1.0)  # Vector{Float64}

# ----------------------------
# Dense channel (length Lh)
# ----------------------------
h_true = randn(ComplexF64, Lh)
h_true ./= norm(h_true)            # normalize power

y_clean = JLDPC.myconv(s, h_true)[1:n]

# AWGN (per complex dimension)
snr_lin = 10^(snr_db/10)
σ = sqrt(1/snr_lin)
noise = σ .* (randn(n) .+ 1im*randn(n)) ./ sqrt(2)
y = y_clean .+ noise

# ----------------------------
# Pilots (~10% of positions)
# ----------------------------
pcount     = max(1, round(Int, 0.1n))
pilot_pos  = pilot_idx[1:min(end, pcount)]
pilot_bpsk = fill(-1.0 + 0im, length(pilot_pos))  # pilots are -1 for all-zero word

# ----------------------------
# Joint decode (dense channel via full tap support)
# ----------------------------
h_pos = collect(1:Lh)  # allow every tap (dense)
xhat, h_est, _ = JLDPC.decode_sparse_joint(
    y, code, idrows, pilot_pos, pilot_bpsk, h_pos;
    λ=0.05, γ=1e-4, max_iter=120, verbose=false
)

# ----------------------------
# Results
# ----------------------------
errs  = sum(Int.(xhat))                              # true bits are all 0
valid = JLDPC.is_valid_codeword(collect(Int, xhat), Hs)

# Compare h_est to zero-padded h_true (same indexing convention as myconv)
corr = abs(sum(conj.(h_true) .* h_est[1:length(h_true)])) /
       (norm(h_true)*norm(h_est[1:length(h_true)]) + eps())

println("valid codeword?  ", valid)
println("errors (/n)      ", errs, " / ", n)
println("‖ĥ‖ (≈1)        ", round(norm(h_est), digits=4))
println("tap corr         ", round(corr, digits=4))
println("ĥ (first Lh)    ", h_est[1:Lh])
