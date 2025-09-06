# JLDPC.jl

Lightweight LDPC utilities for Julia: loading parity-check / generator matrices, encoding, simple belief-prop, and a joint sparse-channel + codeword decoder for quick experiments.

<!-- Badges (uncomment/adjust once you have CI/docs set up)
[![CI](https://github.com/<YOUR_USER>/JLDPC.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/<YOUR_USER>/JLDPC.jl/actions)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://<YOUR_USER>.github.io/JLDPC.jl)
-->

## Features

- Read `.H` / `.pchk` (parity-check) and `.gen` (dense generator) files.
- Path resolver for data files (add folders or use `ENV["LDPC_DATA_DIR"]`).
- Encoding that respects the `.gen` column permutation.
- A simple sum-product (BP) decoder.
- A joint decoder that estimates a **sparse** complex channel and bits together.

## Install

```julia
] add https://github.com/<YOUR_USER>/JLDPC.jl
## Examples

- [`examples/basic_usage.jl`](examples/basic_usage.jl) — clean, runnable walkthrough.  
Run it with:
```bash
julia --project -e 'include("examples/basic_usage.jl")'

