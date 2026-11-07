# # MPSKitTaSK.jl
#
# MPSKitTaSK.jl implements the tangent-space Krylov (TaSK) method for dynamical spectral functions of
# finite matrix product states on top of [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl)
# ([arXiv:2510.07279](https://arxiv.org/abs/2510.07279)). A local (or momentum-resolved) operator applied to the
# ground state is projected onto the quasiparticle tangent space, and a Lanczos recursion of the effective
# excitation Hamiltonian yields the poles and weights of the spectral function. Abelian and non-abelian symmetries
# (e.g. `SU2Irrep`) of TensorKit.jl are supported, as are arbitrary N-site and charged operators.
#
# ## Installation
#
# The package is not yet registered in the Julia general registry. It can be installed through the package manager with the following command:
#
# ```julia-repl
# pkg> add https://github.com/ManyBodyLab/MPSKitTaSK.jl
# ```
#
# ## Code Samples
#
# ```julia
# using MPSKit, MPSKitModels, TensorKit
# using MPSKitTaSK
#
# # ground state of the spin-1/2 Heisenberg chain with SU(2) symmetry
# L = 16
# H = heisenberg_XXX(SU2Irrep, FiniteChain(L); spin = 1 // 2)
# P = SU2Space(1 // 2 => 1)
# Vs = [isodd(b) ? SU2Space(1 // 2 => 2, 3 // 2 => 1) : SU2Space(0 => 2, 1 => 2) for b in 1:(L - 1)]
# gs, envs = find_groundstate(FiniteMPS(fill(P, L), Vs), H, DMRG2(; trunc = truncrank(64), verbosity = 0))
#
# # spin structure factor ∑ₐ Sᵃᵃ(k, ω) at momentum k = π from one TaSK Lanczos run
# S = spin_vector_operator(SU2Irrep)
# res = task_spectrum(H, S, gs, π, TaSK(; krylovdim = 50); envs)
# res.energies, res.weights                      # poles and spectral weights (∑ weights = b0²)
#
# ω = range(0, 3.5; length = 300)
# S_ω = broaden(res, ω; broadening = :gaussian, σ = 0.05)
#
# # full momentum-resolved S(k, ω) for k = 2πn/L
# dsf = dynamical_structure_factor(S, gs, H, 2π .* (0:(L - 1)) ./ L, ω; envs, broadening = :gaussian, sigma = 0.05)
# ```
#
# With `Plots.jl` loaded, `plot_spectral_function(dsf.results, dsf.k_over_pi; ω_grid = ω)` shows the heatmap.
# A complete example, which also compares TaSK against TDVP time evolution in the tangent space, is
# [`examples/heisenberg_finite.jl`](https://github.com/ManyBodyLab/MPSKitTaSK.jl/blob/main/examples/heisenberg_finite.jl):
#
# ```sh
# julia examples/heisenberg_finite.jl   # Julia ≥ 1.11 picks up the local package via [sources]
# ```
#
# ## License
#
# MPSKitTaSK.jl is licensed under the [Apache License 2.0](LICENSE). By using or interacting with this software in any way, you agree to the license of this software.
