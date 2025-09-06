# src/io.jl

# -------------------------------------------------------------------
# Low-level readers (via path resolver)
# -------------------------------------------------------------------

"""
    readsparse(name::AbstractString) -> SparseMatrixCSC{Bool,Int}

Read an LDPC parity-check matrix from `.H` or `.pchk`.
`name` can be a bare stem like "128-256-4" or a full filename.
"""
function readsparse(name::AbstractString)
    path = endswith(lowercase(name), ".h") || endswith(lowercase(name), ".pchk") ?
           resolve_ldpc_file(name) :
           resolve_ldpc_any(name, ["H", "pchk"])

    ii, jj = Int[], Int[]
    for s in eachline(path)
        m = match(r"^ *(\d+):(.+)$", s)
        if m !== nothing
            i  = parse(Int, m[1])
            js = parse.(Int, split(m[2]))
            append!(ii, repeat([i + 1], length(js)))
            append!(jj, js .+ 1)
        end
    end
    return sparse(ii, jj, true)
end

"""
    readgenerator(name::AbstractString) -> (icols, G)

Read an LDPC generator matrix from `.gen`.
`name` can be a bare stem like "128-256-4" or a full filename.
Returns `(icols, G)` where `icols` is the permutation and `G` is a BitMatrix.
"""
function readgenerator(name::AbstractString)
    path = endswith(lowercase(name), ".gen") ? resolve_ldpc_file(name) :
           resolve_ldpc_file(string(name, ".gen"))

    open(path) do io
        read(io, UInt32) == 0x00004780 || error("Bad generator")
        read(io, UInt8)  == 0x64       || error("Bad generator: must be dense")
        p   = Int(read(io, UInt32))
        n   = Int(read(io, UInt32))
        icols = [Int(read(io, UInt32)) + 1 for _ in 1:n] |> invperm
        Int(read(io, UInt32)) == p     || error("Bad row size")
        Int(read(io, UInt32)) == n - p || error("Bad column size")
        G = mapreduce(hcat, 1:n - p) do _
            v = [read(io, UInt32) for _ in 1:ceil(Int, p / 32)]
            isodd(length(v)) && push!(v, 0)
            b = BitArray(undef, p)
            b.chunks .= reinterpret(UInt64, v)
            b
        end
        return icols, G
    end
end

# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

"""
    FEC_create_code(k, n, npc; H_path=nothing) -> Code

Construct a `Code` by loading "<k>-<n>-<npc>.H" or ".pchk" from the search paths.
Pass `H_path` to override with a specific file path or name.
"""
function FEC_create_code(k::Int, n::Int, npc::Int; H_path::Union{Nothing,String}=nothing)
    stem_or_name = H_path === nothing ? "$(k)-$(n)-$(npc)" : H_path
    Hs = readsparse(stem_or_name)
    return Code(k, n, npc, nothing, nothing, collect(Hs))
end

"""
    encode(code, bits; gen_path=nothing) -> Vector{Int}

Encode `bits` using the matching dense ".gen" file named "<k>-<n>-<npc>.gen".
Pass `gen_path` to override with a specific file path or name.
"""
function encode(ldpc::Code, bits::AbstractVector{<:Integer}; gen_path::Union{Nothing,String}=nothing)
    length(bits) == ldpc.k || throw(ArgumentError("Wrong bit length: got $(length(bits)), expected $(ldpc.k)"))
    if ldpc.icols === nothing
        name = gen_path === nothing ? "$(ldpc.k)-$(ldpc.n)-$(ldpc.npc)" : gen_path
        ldpc.icols, ldpc.gen = readgenerator(name)
    end
    @assert ldpc.gen !== nothing
    parity = map(eachrow(ldpc.gen)) do g
        reduce(⊻, g .* bits)
    end |> BitVector
    return vcat(parity, bits)[ldpc.icols]
end
