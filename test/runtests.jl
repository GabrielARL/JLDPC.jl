using Test, JLDPC, SparseArrays, Random, LinearAlgebra

# -------------------------------------------------------------------
# Search path bootstrap (Pkg.test runs in a temp dir!)
# -------------------------------------------------------------------
try
    JLDPC.clear_ldpc_paths!()
catch
end

# Prefer package data + test fixtures, then ENV, then CWD
try
    JLDPC.add_ldpc_path!(joinpath(pkgdir(JLDPC), "data"))
    JLDPC.add_ldpc_path!(joinpath(pkgdir(JLDPC), "test", "data"))  # <— NEW
catch
end
if haskey(ENV, "LDPC_DATA_DIR")
    JLDPC.add_ldpc_path!(abspath(ENV["LDPC_DATA_DIR"]))
end
JLDPC.add_ldpc_path!(pwd())

@info "LDPC search paths" paths=JLDPC.get_ldpc_paths()

# -------------------------------------------------------------------
# Helpers for discovery
# -------------------------------------------------------------------
# Probe using the public readers so we accept either .H or .pchk, and .gen via stem
has_parity(stem::AbstractString) = try; JLDPC.readsparse(stem); true; catch; false; end
has_gen(stem::AbstractString)    = try; JLDPC.readgenerator(stem); true; catch; false; end

function find_ldpc_stems()
    stems = Set{NTuple{3,Int}}()
    for d in JLDPC.get_ldpc_paths()
        isdir(d) || continue
        for f in readdir(d)
            m = match(r"^(\d+)-(\d+)-(\d+)\.(H|pchk|gen)$", f)
            m === nothing && continue
            k = parse(Int, m.captures[1])
            n = parse(Int, m.captures[2])
            npc = parse(Int, m.captures[3])
            push!(stems, (k,n,npc))
        end
    end
    sort!(collect(stems))
end

# Require both parity (H or pchk) and gen; use the stem to resolve
function complete_stems(stems)
    filter(stems) do (k,n,npc)
        stem = "$(k)-$(n)-$(npc)"
        has_parity(stem) && has_gen(stem)
    end
end

const ALL_STEMS = complete_stems(find_ldpc_stems())
const MISSING   = setdiff(find_ldpc_stems(), ALL_STEMS)

@testset "Discovered code families" begin
    @test !isempty(ALL_STEMS)
    for t in MISSING
        @info "Skipping incomplete code family (missing .gen or .H/.pchk)" triple=t
    end
end

# -------------------------------------------------------------------
# Original unit tests
# -------------------------------------------------------------------
@testset "Hardcoded H (64,128,4)" begin
    code, cols, idrows, pilots = JLDPC.initcode(64,128,4)
    @test size(code.H) == (64,128)
    Hs = JLDPC.get_H_sparse(code)
    x0 = zeros(Int, 128)
    @test JLDPC.is_valid_codeword(x0, Hs)
    x1 = copy(x0); x1[1] = 1
    @test !JLDPC.is_valid_codeword(x1, Hs)
end

@testset "myconv vs reference" begin
    x = [1.0, -2.0, 3.0]
    h = [0.5, -1.0]
    y_ref = [x[1]*h[1],
             x[1]*h[2]+x[2]*h[1],
             x[2]*h[2]+x[3]*h[1],
             x[3]*h[2]]
    y = JLDPC.myconv(x,h)
    @test isapprox(y, y_ref; atol=1e-12)
end

@testset "prefix_suffix_products sanity" begin
    x = rand(10)
    prefix, suffix = JLDPC.prefix_suffix_products(x)
    total = prod(x)
    for i in 1:length(x)
        @test isapprox(prefix[i]*x[i]*suffix[i], total; rtol=1e-12, atol=1e-12)
    end
end

@testset "AWGN hard decision (all-zero codeword, no decoder)" begin
    code, _, _, _ = JLDPC.initcode(64,128,4)
    n = code.n
    s = fill(-1.0, n)
    σ = 0.2
    for _ in 1:3
        y = s .+ σ .* randn(n)
        xhat = @. Int(y > 0)
        @test sum(xhat) == 0
    end
end

@testset "Joint decoder (single-tap channel + noise)" begin
    Random.seed!(202)
    code, cols, idrows, pilot_idx = JLDPC.initcode(64,128,4)
    n = code.n
    Hs = JLDPC.get_H_sparse(code)
    x_bpsk = fill(-1.0 + 0im, n)
    h_pos = [1]
    h_true = ComplexF64[1.0]
    y_clean = JLDPC.myconv(x_bpsk, h_true)[1:n]
    σ = 0.05
    noise = σ .* (randn(n) .+ 1im*randn(n)) ./ sqrt(2)
    y = y_clean .+ noise
    pcount = max(1, round(Int, 0.1n))
    pilot_pos = pilot_idx[1:min(end, pcount)]
    pilot_bpsk = fill(-1.0 + 0im, length(pilot_pos))
    xhat, h_est, _ = JLDPC.decode_sparse_joint(y, code, idrows, pilot_pos, pilot_bpsk, h_pos;
                                               λ=0.1, γ=1e-4, max_iter=60, verbose=false)
    @test JLDPC.is_valid_codeword(collect(Int, xhat), Hs)
    @test sum(xhat) == 0
    @test length(h_est) == 1
    @test isapprox(abs(h_est[1]), 1.0; atol=0.1)
end

@testset "Joint decoder (3-tap sparse channel + noise)" begin
    Random.seed!(303)
    code, cols, idrows, pilot_idx = JLDPC.initcode(64,128,4)
    n = code.n
    Hs = JLDPC.get_H_sparse(code)
    s = fill(-1.0 + 0im, n)
    h_pos = [1, 7, 13]
    h_vals = [1.0 + 0im, 0.5*cis(0.7), 0.3*cis(-0.4)]
    h_true = zeros(ComplexF64, n); h_true[h_pos] .= h_vals
    h_true ./= norm(h_true)
    y_clean = JLDPC.myconv(s, h_true)[1:n]
    σ = 0.04
    noise = σ .* (randn(n) .+ 1im*randn(n)) ./ sqrt(2)
    y = y_clean .+ noise
    pcount = max(1, round(Int, 0.1n))
    pilot_pos = pilot_idx[1:min(end, pcount)]
    pilot_bpsk = fill(-1.0 + 0im, length(pilot_pos))
    xhat, h_est, _ = JLDPC.decode_sparse_joint(y, code, idrows, pilot_pos, pilot_bpsk, h_pos;
                                               λ=0.05, γ=1e-4, max_iter=100, verbose=false)
    @test JLDPC.is_valid_codeword(Int.(xhat), Hs)
    @test sum(xhat) ≤ 1
    @test isapprox(norm(h_est), 1.0; atol=0.15)
end

# -------------------------------------------------------------------
# Parameterized tests across all complete stems
# -------------------------------------------------------------------
@testset "Load + encode + parity-check for all codes" begin
    for (k,n,npc) in ALL_STEMS
        @testset "($(k),$(n),$(npc))" begin
            code, cols, idrows, _ = JLDPC.initcode(k,n,npc)
            @test size(code.H) == (n - k, n)
            bits = rand(0:1, k)
            enc  = JLDPC.encode(code, bits)
            Hs   = JLDPC.get_H_sparse(code)
            @test length(enc) == n
            @test JLDPC.is_valid_codeword(collect(Int, enc), Hs)
        end
    end
end
