# examples/longcode_awgn_decode.jl
#
# Demonstrate LDPC(512,1024,4) decoding in AWGN
# using belief propagation.

using JLDPC, Random, LinearAlgebra

# 1) Make sure the package can find the matrices
try JLDPC.add_ldpc_path!(joinpath(@__DIR__, "..", "data")) catch end

# 2) Build the (512,1024,4) code
code, col_indices, parity_indices, _ = JLDPC.initcode(512, 1024, 4)
Hs = JLDPC.get_H_sparse(code)
k, n = code.k, code.n
println("Loaded code: ", code, "  (H size = ", size(code.H), ")")

# 3) All-zero message (simplest baseline)
msg = zeros(Int, k)
cw  = JLDPC.encode(code, msg)
println("Parity check OK? ", JLDPC.is_valid_codeword(Hs, BitVector(cw)))

# 4) BPSK mapping (1 → -1, 0 → +1)
bpsk = @. ifelse(cw == 1, -1.0, +1.0)

# 5) Transmit through AWGN
σ = 0.15
y = bpsk .+ σ .* randn(n)

# 6) Hard decision baseline
xhat_hard = @. y < 0
errs_hard = sum(xhat_hard .!= BitVector(cw))
println("Hard decision errors: $errs_hard / $n")

# 7) Belief propagation decoding
xhat_bp, iters = JLDPC.sum_product_decode(
    Hs, y, σ^2, parity_indices, col_indices; max_iter=80
)
errs_bp  = sum((xhat_bp .== 1) .!= BitVector(cw))
valid_bp = JLDPC.is_valid_codeword(Hs, BitVector(xhat_bp .== 1))

println("BP iterations: ", iters)
println("BP errors    : $errs_bp / $n")
println("BP valid?    : ", valid_bp)
