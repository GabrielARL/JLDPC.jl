module JLDPC

using SparseArrays, GaloisFields, Random, LinearAlgebra
using Optim, LineSearches
using StatsBase, SignalAnalysis

# -------------------------------------------------------------------
# Path resolver utilities (add_ldpc_path!, resolve_ldpc_file, etc.)
# -------------------------------------------------------------------
include("paths.jl")

# -------------------------------------------------------------------
# Types
# -------------------------------------------------------------------
mutable struct Code
    k::Int
    n::Int
    npc::Int
    icols::Union{Nothing, Vector{Int}}
    gen::Union{Nothing, BitMatrix}
    H::Matrix{Bool}
end

Base.show(io::IO, ldpc::Code) = print(io, "LDPC($(ldpc.k)/$(ldpc.n))")

# -------------------------------------------------------------------
# Finite field
# -------------------------------------------------------------------
const GF2 = GaloisField(2)

# -------------------------------------------------------------------
# IO (readsparse/readgenerator/FEC_create_code/encode) — needs Code
# -------------------------------------------------------------------
include("io.jl")

# -------------------------------------------------------------------
# Exports
# -------------------------------------------------------------------
export Code,
       # from io.jl:
       FEC_create_code, encode, readsparse, readgenerator,
       # path helpers:
       add_ldpc_path!, remove_ldpc_path!, clear_ldpc_paths!,
       get_ldpc_paths, resolve_ldpc_file, resolve_ldpc_any,
       # core ops & utils:
       initcode, get_H_sparse, is_valid_codeword,
       sum_product_decode, decode_sparse_joint,  # add decode_dense_joint if you have it
       myconv, generate_sparse_channel, modulate, demodulate,
       estimate_channel_from_pilots, resolve_sign_flip,
       prefix_suffix_products, makepacket

# -------------------------------------------------------------------
# Module init
# -------------------------------------------------------------------
function __init__()
    __init_ldpc_paths__()   # defined in paths.jl
end

# -------------------------------------------------------------------
# Mod/demod
# -------------------------------------------------------------------
modulate(x; θ=0.0)   = x == 1 ? cis(θ) : -cis(θ)
demodulate(x; θ=0.0) = x == 1 ? cis(θ) : -cis(θ)

# -------------------------------------------------------------------
# Cache helpers
# -------------------------------------------------------------------
const H_sparse_cache = IdDict{Code, SparseMatrixCSC{Bool, Int}}()
get_H_sparse(code::Code) = get!(H_sparse_cache, code) do
    sparse(code.H)
end

# -------------------------------------------------------------------
# Index helpers
# -------------------------------------------------------------------
function get_row_column_positions(idx::Vector{CartesianIndex{2}}, num_rows::Int)
    rowcols = [Int[] for _ in 1:num_rows]
    for ij in idx
        push!(rowcols[ij[1]], ij[2])
    end
    rowcols
end

function initcode(d_nodes::Int, t_nodes::Int, npc::Int; pilot_row_fraction::Float64=0.1)
    code = FEC_create_code(d_nodes, t_nodes, npc)
    idx = findall(!iszero, code.H)
    num_rows, num_cols = size(code.H)
    idrows = get_row_column_positions(idx, num_rows)
    idx_colwise = findall(!iszero, code.H')
    cols = get_row_column_positions(idx_colwise, num_cols)
    num_parity_rows = num_rows
    start_row = round(Int, (1.0 - pilot_row_fraction) * num_parity_rows) + 1
    pilot_rows = idrows[start_row:end]
    pilot_indices = sort(unique(vcat(pilot_rows...)))
    return code, cols, idrows, pilot_indices
end

# -------------------------------------------------------------------
# Channels & conv
# -------------------------------------------------------------------
function generate_sparse_channel(L_h::Int, sparsity::Int)
    h = zeros(ComplexF64, L_h)
    pos = sample(1:L_h, sparsity; replace=false)
    h[pos] .= randn(ComplexF64, sparsity)
    return h ./ norm(h)
end

function myconv(x::Vector{<:Number}, h::Vector{<:Number})
    n, L = length(x), length(h)
    [sum(@inbounds h[j] * x[i - j + 1] for j in 1:L if 1 <= i - j + 1 <= n) for i in 1:(n + L - 1)]
end

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
function prefix_suffix_products(x::Vector{Float64})
    n = length(x)
    prefix = ones(Float64, n)
    suffix = ones(Float64, n)
    for i in 2:n
        @inbounds prefix[i] = prefix[i - 1] * x[i - 1]
    end
    for i in (n - 1):-1:1
        @inbounds suffix[i] = suffix[i + 1] * x[i + 1]
    end
    return prefix, suffix
end

function estimate_channel_from_pilots(y::Vector{ComplexF64}, pilot_pos::Vector{Int},
                                      pilot_bpsk::Vector{ComplexF64}, L_h::Int)
    X = zeros(ComplexF64, length(pilot_pos), L_h)
    for (i, p) in enumerate(pilot_pos)
        for j in 1:L_h
            if p - j + 1 ∈ 1:length(pilot_bpsk)
                @inbounds X[i, j] = pilot_bpsk[p - j + 1]
            end
        end
    end
    X \ y[pilot_pos]
end

function resolve_sign_flip(x̂::BitVector, z::Vector{Float64}, pilot::Vector{Int},
                           pilot_bpsk::Vector{ComplexF64}, H::SparseMatrixCSC{Bool, Int})
    vote = sum(real(z[pilot]) .* real(pilot_bpsk))
    x̂_flipped = .!x̂
    H_GF = convert(SparseMatrixCSC{GF2, Int}, H)
    x̂_GF         = GF2.(x̂)
    x̂_flipped_GF = GF2.(x̂_flipped)
    syndrome_raw     = count(!iszero, H_GF * x̂_GF)
    syndrome_flipped = count(!iszero, H_GF * x̂_flipped_GF)
    (syndrome_flipped < syndrome_raw || vote < 0) ? x̂_flipped : x̂
end

# Robust GF2 check against Bool sparse matrices
function is_valid_codeword(bits::Vector{Int}, H::SparseMatrixCSC)
    x  = GF2.(bits)
    HG = convert(SparseMatrixCSC{GF2, Int}, H)
    all(HG * x .== GF2(0))
end
is_valid_codeword(H::SparseMatrixCSC{Bool, Int}, x::BitVector) =
    is_valid_codeword(collect(Int, x), H)

# -------------------------------------------------------------------
# Joint sparse decoder (existing)
# -------------------------------------------------------------------
function decode_sparse_joint(y, code::Code, parity_indices, pilot_pos, pilot_bpsk, h_pos;
    λ=1.0, γ=1e-3, η=1.0, h_init=nothing, max_iter=80, verbose=false)

    n = length(y)
    h_prior = h_init === nothing ? randn(ComplexF64, length(h_pos)) : h_init
    θ0 = vcat(zeros(n), real(h_prior), imag(h_prior))

    function loss_and_grad!(g, θ)
        z   = @view θ[1:n]
        h_r = @view θ[n+1:n+length(h_pos)]
        h_i = @view θ[n+length(h_pos)+1:end]
        h_vals = ComplexF64.(h_r, h_i)

        x = tanh.(z)
        h_full = zeros(ComplexF64, n)
        @inbounds h_full[h_pos] .= h_vals

        yhat_full  = myconv(x, h_full)
        y_obs_full = vcat(y, zeros(n-1))
        res_full   = yhat_full .- y_obs_full

        hr     = reverse(conj.(h_full))
        d_full = myconv(res_full, hr)
        dLdx   = 2 .* real.(@view d_full[n:2n-1])

        sech2 = 1 .- x.^2
        @inbounds @. g[1:n] = dLdx * sech2 + 2γ * z

        for (ii, j) in enumerate(h_pos)
            x_pad = vcat(zeros(j-1), x, zeros(n - j))
            g_c   = 2 * sum(res_full .* conj.(x_pad))
            g[n + ii]                 =  real(g_c) + 2γ * h_r[ii]
            g[n + length(h_pos) + ii] =  imag(g_c) + 2γ * h_i[ii]
        end

        parity_loss = 0.0
        for inds in parity_indices
            p = prod(@view x[inds])
            parity_loss += (1 - p)^2
            c = 2 * (1 - p)
            for i in inds
                g[i] += λ * (-c * p / x[i]) * (1 - x[i]^2)
            end
        end

        return sum(abs2, res_full) + λ * parity_loss + γ * (sum(abs2, z) + sum(abs2, h_r + h_i * im))
    end

    opt = Optim.Options(f_abstol=1e-6, g_abstol=1e-7, iterations=max_iter)
    result = Optim.optimize(x -> loss_and_grad!(zeros(length(θ0)), x), θ0, Optim.LBFGS(), opt)

    θ_opt = Optim.minimizer(result)
    z_opt = θ_opt[1:n]
    h_est = ComplexF64.(θ_opt[n+1:n+length(h_pos)], θ_opt[n+length(h_pos)+1:end])

    x̂_raw = tanh.(z_opt) .> 0
    x̂ = resolve_sign_flip(x̂_raw, z_opt, pilot_pos, pilot_bpsk, get_H_sparse(code))

    if verbose
        println("\n📦 Optimization complete.")
        println("✅ Valid codeword: ", is_valid_codeword(get_H_sparse(code), BitVector(x̂)))
        println("📡 Estimated h: ", h_est)
    end

    return x̂, h_est ./ (norm(h_est) > 0 ? norm(h_est) : 1), result
end

# -------------------------------------------------------------------
# Belief Propagation (sum-product)
# -------------------------------------------------------------------
function sum_product_decode(H::SparseMatrixCSC{Bool, Int},
                            y::Vector{Float64},
                            σ²::Float64,
                            parity_indices::Vector{Vector{Int}},
                            col_indices::Vector{Vector{Int}};
                            max_iter::Int=50)

    m, n = size(H)
    L_ch = @. 2 * y / σ²
    M = Dict{Tuple{Int, Int}, Float64}()

    for j in 1:n
        for i in col_indices[j]
            M[(i, j)] = L_ch[j]
        end
    end

    for iter in 1:max_iter
        # C→V
        for i in 1:m
            neighbors = parity_indices[i]
            d = length(neighbors)
            tanhs = [tanh(0.5 * M[(i, j)]) for j in neighbors]
            prefix = ones(Float64, d)
            suffix = ones(Float64, d)
            for j in 2:d; prefix[j] = prefix[j-1] * tanhs[j-1]; end
            for j in (d-1):-1:1; suffix[j] = suffix[j+1] * tanhs[j+1]; end
            for j in 1:d
                prod_except = prefix[j] * suffix[j]
                M[(i, neighbors[j])] = 2 * atanh(clamp(prod_except, -0.999999, 0.999999))
            end
        end

        # V→C
        for j in 1:n
            neighbors = col_indices[j]
            for i in neighbors
                msg = L_ch[j] + sum(M[(k, j)] for k in setdiff(neighbors, i))
                M[(i, j)] = msg
            end
        end

        # early stop
        L_post = zeros(Float64, n)
        for j in 1:n
            L_post[j] = L_ch[j] + sum(M[(i, j)] for i in col_indices[j])
        end
        x_hat = @. Int(L_post < 0)
        if is_valid_codeword(x_hat, H)
            return x_hat, iter
        end
    end

    L_post = zeros(Float64, n)
    for j in 1:n
        L_post[j] = L_ch[j] + sum(M[(i, j)] for i in col_indices[j])
    end
    return @. Int(L_post < 0), max_iter
end

# -------------------------------------------------------------------
# Packet maker (helper)
# -------------------------------------------------------------------
function makepacket(code::Code, num_train::Int, num_data::Int, gap::Int)
    k, n = code.k, code.n
    packet = Float64[]
    x_train = repeat(mseq(8), num_train)
    append!(packet, x_train)
    packet_gap = fill(0.0, gap)
    append!(packet, packet_gap)
    x_datas = zeros(Float64, num_data, n)
    d_datas = zeros(Float64, num_data, k)
    for i in 1:num_data
        bseq   = mseq(11)[i : k+i-1]
        d_test = Int.((bseq .+ 1) ./ 2)
        E_data = encode(code, d_test)
        x_data = modulate.(E_data)
        x_datas[i, :] = x_data
        d_datas[i, :] = d_test
        append!(packet, x_data)
        if i != num_data
            append!(packet, packet_gap)
        end
    end
    return packet, x_datas, d_datas
end

end # module
