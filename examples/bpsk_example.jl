using Random
using JLDPC
using JLDPC.CFO_est
using DSP: conv

# ----- system -----
fs   = 8_000.0
pbfs = 192_000.0
sps  = Int(pbfs ÷ fs); @assert sps*fs == pbfs
fc   = 24_000.0
β    = 0.25
span = 6
taps = rrc(β, sps, span)
L    = length(taps)

# ----- frame -----
Np  = 2048          # pilot
Nd  = 8192          # payload
xb  = [bits_pm1(Np); bits_pm1(Nd)]

# ----- channel -----
paths = [(0.0, 0.0), (0.7e-3, -3.0), (1.6e-3, -6.0)]
h    = chpb(paths, pbfs)

# ----- tx / channel / rx -----
tx   = up(xb, fs, sps, fc, taps)
ypb  = conv(tx, h)
zall, symdel, _ = down(ypb, pbfs, sps, fc, taps)

# coarse timing
z0, soff = align(zall, xb, Np, symdel)

# pilot-aided CFO (guarded)
fhat, dphi, ok = cfo1(z0[1:Np], xb[1:Np], fs)
z1 = ok ? derot(z0, dphi) : z0

# Eb/N0 noise at symbol rate
ebn0 = 10.0
zn   = awgn(z1, ebn0)

# DFE (RLS) equalization
train = complex.(xb[1:Np])
y_eq  = dfe(zn, train, length(xb); ff=40, fb=40)

# BER on payload
ber = berp(y_eq, xb, Np)

println("JLDPC example: BPSK pilot-aided CFO + DFE @ Eb/N0 = $(ebn0) dB")
println("• sps=$(sps), span=$(span), L=$(L), symdelay=$(symdel), soff=$(soff)")
println("• CFO_est ≈ $(round(fhat, digits=4)) Hz (applied=$(ok))")
println("• DFE(ff=40, fb=40), train=$(Np) (RLS)")
println("• BER(payload) = $(ber)")
