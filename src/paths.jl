# src/paths.jl

# ordered list of directories we’ll search
const _LDPC_SEARCH_PATHS = Ref{Vector{String}}(String[])

# build defaults: data/, pkg root, test/data, ENV var, pwd()
function _default_ldpc_paths()
    paths = String[]
    try
        pkgd = pkgdir(@__MODULE__)
        push!(paths, joinpath(pkgd, "data"))            # <— as before
        push!(paths, pkgd)                              # <— as before
        push!(paths, joinpath(pkgd, "test", "data"))    # <— NEW: so Pkg.test() finds fixtures
    catch
    end
    if haskey(ENV, "LDPC_DATA_DIR")
        push!(paths, abspath(ENV["LDPC_DATA_DIR"]))
    end
    push!(paths, pwd())
    return unique(normpath.(paths))
end

# called from module __init__()
function __init_ldpc_paths__()
    _LDPC_SEARCH_PATHS[] = _default_ldpc_paths()
end

get_ldpc_paths() = copy(_LDPC_SEARCH_PATHS[])

function add_ldpc_path!(dir::AbstractString)
    pushfirst!(_LDPC_SEARCH_PATHS[], abspath(dir))
    unique!(_LDPC_SEARCH_PATHS[])
    get_ldpc_paths()
end

function remove_ldpc_path!(dir::AbstractString)
    filter!(p -> p != abspath(dir), _LDPC_SEARCH_PATHS[])
    get_ldpc_paths()
end

# Previously this cleared to []; that makes resolution impossible until you add paths back.
# Minimal ergonomic change: reset to defaults instead.
clear_ldpc_paths!() = (_LDPC_SEARCH_PATHS[] = _default_ldpc_paths())

function resolve_ldpc_file(name_or_path::AbstractString; must_exist::Bool=true)
    isfile(name_or_path) && return abspath(name_or_path)
    for d in _LDPC_SEARCH_PATHS[]
        p = joinpath(d, name_or_path)
        if isfile(p)
            return abspath(p)
        end
    end
    if must_exist
        error("Could not find file $(name_or_path). Searched: " * join(_LDPC_SEARCH_PATHS[], " | "))
    else
        return abspath(name_or_path)
    end
end

function resolve_ldpc_any(stem::AbstractString, exts::Vector{String})
    for ext in exts
        name = endswith(stem, "." * ext) ? stem : string(stem, ".", ext)
        try
            return resolve_ldpc_file(name)
        catch
        end
    end
    error("Could not find any of: " * join(["$(stem).$(ext)" for ext in exts], ", "))
end
