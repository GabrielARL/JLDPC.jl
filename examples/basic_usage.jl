using JLDPC, Random, LinearAlgebra

# Init a (64,128,4) code
code, cols, idrows, pilots = JLDPC.initcode(64,128,4)
Hs = JLDPC.get_H_sparse(code)
println("Loaded code: ", code, "  (H size = ", size(code.H), ")")

# Random message
message = rand(0:1, code.k)

# Encode
codeword = collect(Int, JLDPC.encode(code, message))
if JLDPC.is_valid_codeword(codeword, Hs)
    println("Parity check OK ✓")
end

# Transmit through AWGN
σ = 0.2
bpsk = [b == 1 ? 1.0 : -1.0 for b in codeword]
rx   = bpsk .+ σ .* randn(code.n)

# Hard-decision decode
hardbits = @. Int(rx > 0)
println("Hard decision bit errors (AWGN): ", sum(hardbits .!= codeword))
