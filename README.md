# JLDPC.jl

[![Build Status](https://github.com/GabrielARL/JLDPC.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/GabrielARL/JLDPC.jl/actions)  
*A Julia package for LDPC coding, modulation, and joint sparse/dense decoding in noisy and multipath channels.*

---

## ✨ Features
- Load and manage LDPC codes (`.H`, `.pchk`, `.gen`) with automatic path resolution.
- Encode/decode messages with **LDPC codes**.
- Simulate transmission over **AWGN** and **multipath channels**.
- **Sum-product (BP) decoder** for classical LDPC decoding.
- **Joint sparse channel decoding** (gradient-based) for multipath fading channels.
- Experimental support for **dense joint channel decoding**.

---

## 🚀 Installation
Clone the repository and activate it in Julia:
```julia
using Pkg
Pkg.clone("https://github.com/GabrielARL/JLDPC.jl.git")

