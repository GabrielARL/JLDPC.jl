# examples/awgn_decode.jl
# AWGN decoding demo:
# 1) load code & indices
# 2) encode random message
# 3) transmit over AWGN
# 4) hard decision + sum-product (BP) decoding

using JLDPC, Random, LinearAlgebra, SparseArrays

# (optional) ensure repo's data/ is on the search path
try
    JLDPC.add_ldpc_path!(joinpath(@__DIR__, "..", "data"))
catch
end

# --- 1) code & parity structure
code, col_indices, parity_indices, _ = JLDPC.initcode(64, 128, 4)
Hs = JLDPC.get_H_sparse(code)
k, n = code.k, code.n
println("Loaded code: ", code, "  (H size = ", size(code.H), ")")

# --- 2) encode a random message
Random.seed!(123)
msg  = rand(0:1, k)
cw   = JLDPC.encode(code, msg)                    # BitVector
valid = JLDPC.is_valid_codeword(Hs, BitVector(cw))  # H first, bits second
println("Parity check OK? ", valid)

# --- 3) BPSK + AWGN
# Use the package's convention: 1 ↦ -1.0, 0 ↦ +1.0
s = @. ifelse(cw == 1, -1.0, +1.0)     # <<< CHANGED

σ = 0.25
y = s .+ σ .* randn(n)

# --- 4a) hard decision baseline
# Decide 1 when sample is NEGATIVE (since 1 ↦ -1)
xhat_hard = @. y < 0                    # <<< CHANGED
errs_hard = sum(xhat_hard .!= BitVector(cw))
println("Hard decision errors: $errs_hard / $n")

# --- 4b) Sum-Product (BP) decoding (unchanged)
xhat_bp, iters = JLDPC.sum_product_decode(Hs, y, σ^2, parity_indices, col_indices; max_iter=50)
errs_bp = sum((xhat_bp .== 1) .!= BitVector(cw))
valid_bp = JLDPC.is_valid_codeword(Hs, BitVector(xhat_bp .== 1))
println("BP iterations: $iters")
println("BP errors    : $errs_bp / $n")
println("BP valid?    : $valid_bp")

