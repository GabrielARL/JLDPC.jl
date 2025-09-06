# examples/multipath_joint_decode_random.jl
using JLDPC, Random, LinearAlgebra

# 1) Load code + indices
try JLDPC.add_ldpc_path!(joinpath(@__DIR__, "..", "data")) catch end
code, col_indices, parity_indices, pilot_idx = JLDPC.initcode(64, 128, 4)
Hs = JLDPC.get_H_sparse(code)
k, n = code.k, code.n
println("Loaded code: ", code, "  (H size = ", size(code.H), ")")

# 2) Message -> codeword -> BPSK (package convention: 1 -> -1.0, 0 -> +1.0)
Random.seed!(42)
msg = rand(0:1, k)
cw  = JLDPC.encode(code, msg)                       # Vector{Int} or BitVector
true_bits = BitVector(cw)                          # normalize
bpsk_full = @. ifelse(cw == 1, -1.0 + 0im, +1.0 + 0im)

# 3) Sparse multipath channel
h_pos  = [1, 7, 13]
h_vals = [1.0 + 0im, 0.5*cis(0.7), 0.3*cis(-0.4)]
h_full = zeros(ComplexF64, n); h_full[h_pos] .= h_vals; h_full ./= norm(h_full)

# 4) Transmit + noise
y_clean = JLDPC.myconv(bpsk_full, h_full)[1:n]
σ       = 0.06
noise   = σ .* (randn(n) .+ 1im*randn(n)) ./ sqrt(2)
y       = y_clean .+ noise

# 5) Pick safer pilot positions (avoid first Lh-1 to reduce edge effects)
Lh = maximum(h_pos)
candidate = collect(Lh:n)                          # avoid indices < Lh
pcount    = max(8, round(Int, 0.15n))              # ~15% pilots; up a bit for stability
pilot_pos = candidate[1:min(length(candidate), pcount)]
pilot_bpsk = ComplexF64.(bpsk_full[pilot_pos])     # SUBSET for sign-flip vote

# 6) Estimate initial channel from pilots (needs FULL BPSK)
h_ls_full = JLDPC.estimate_channel_from_pilots(y, pilot_pos, bpsk_full, Lh)
# keep only the taps we model (sparse at h_pos); guard for short LS vectors
h_init = ComplexF64[ (i <= length(h_ls_full) ? h_ls_full[i] : 0.0 + 0im) for i in h_pos ]
if norm(h_init) > 0
    h_init ./= norm(h_init)
else
    # fallback if LS is degenerate
    h_init .= 0.0 + 0im
    h_init[1] = 1.0 + 0im
end

# 7) Joint decode (slightly stronger parity penalty + more iters)
xhat, hhat, _ = JLDPC.decode_sparse_joint(
    y, code, parity_indices, pilot_pos, pilot_bpsk, h_pos;
    λ=0.15, γ=1e-4, h_init=h_init, max_iter=200, verbose=false
)

# 8) Metrics
valid = JLDPC.is_valid_codeword(Hs, BitVector(xhat))
ber   = sum(xhat .⊻ true_bits) / n
println("valid codeword?  ", valid)
println("‖ĥ‖ (≈1)        ", round(norm(hhat), digits=4))
println("BER (vs true)    ", round(ber, digits=6))
println("ĥ (tap-order)   ", hhat)
