#!/usr/bin/env julia
using Random, LinearAlgebra
using JLDPC

# ----------------------------
# Config
# ----------------------------
Random.seed!(123)
stems   = [(128,256,4), (512,1024,4)]   # try longer codes here
Lh      = 15                             # dense channel length
snr_db  = 12.0                           # SNR (dB)
λ       = 0.05                           # parity penalty weight
γ       = 1e-4                           # L2 regularization
maxit   = 200                            # more iterations for longer codes

# ----------------------------
# Helpers
# ----------------------------
hasfile(name) = try; JLDPC.resolve_ldpc_file(name); true; catch; false; end

function try_decode(k, n, npc; Lh=15, snr_db=12.0, λ=0.05, γ=1e-4, maxit=200)
    println("\n=== Code ($k,$n,$npc) ===")
    # Sanity: require both files to exist on the LDPC search path(s)
    stem = "$(k)-$(n)-$(npc)"
    if !(hasfile(stem*".H") && hasfile(stem*".gen"))
        println("…skipping: missing $stem.(H|gen) on search paths: ", JLDPC.get_ldpc_paths())
        return
    end

    # Load code + structure
    code, cols, idrows, pilot_idx = JLDPC.initcode(k, n, npc)
    Hs = JLDPC.get_H_sparse(code)
    println("Loaded code: $(code)  (H size = ", size(code.H), ")")
    println("Parity check OK? ", JLDPC.is_valid_codeword(zeros(Int,n), Hs))

    # All-zero message -> encode -> BPSK (-1 for 0, +1 for 1)
    bits  = zeros(Int, k)
    x_enc = JLDPC.encode(code, bits)
    s     = ifelse.(x_enc .== 1, 1.0, -1.0)

    # Dense channel
    h_true = randn(ComplexF64, Lh); h_true ./= norm(h_true)
    y_clean = JLDPC.myconv(s, h_true)[1:n]

    # AWGN
    snr_lin = 10^(snr_db/10)
    σ = sqrt(1/snr_lin)
    noise = σ .* (randn(n) .+ 1im*randn(n)) ./ sqrt(2)
    y = y_clean .+ noise

    # Pilots (~10%)
    pcount     = max(1, round(Int, 0.1n))
    pilot_pos  = pilot_idx[1:min(end, pcount)]
    pilot_bpsk = fill(-1.0 + 0im, length(pilot_pos))

    # Joint decode with dense support
    h_pos = collect(1:Lh)
    xhat, h_est, _ = JLDPC.decode_sparse_joint(
        y, code, idrows, pilot_pos, pilot_bpsk, h_pos;
        λ=λ, γ=γ, max_iter=maxit, verbose=false
    )

    # Results
    errs  = sum(Int.(xhat))  # ground truth all zeros
    valid = JLDPC.is_valid_codeword(collect(Int, xhat), Hs)

    # channel correlation (first Lh taps)
    corr = abs(sum(conj.(h_true) .* h_est[1:length(h_true)])) /
           (norm(h_true)*norm(h_est[1:length(h_true)]) + eps())

    println("valid codeword?  ", valid)
    println("errors (/n)      ", errs, " / ", n)
    println("‖ĥ‖ (≈1)        ", round(norm(h_est), digits=4))
    println("tap corr         ", round(corr, digits=4))
    println("ĥ (first Lh)    ", h_est[1:Lh])
end

# ----------------------------
# Run over stems
# ----------------------------
for (k,n,npc) in stems
    try_decode(k, n, npc; Lh=Lh, snr_db=snr_db, λ=λ, γ=γ, maxit=maxit)
end
