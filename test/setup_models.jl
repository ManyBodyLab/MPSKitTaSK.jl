# Tiny models with exactly representable ground states, shared by the test files.
# Exactness makes the TaSK spectra independent of the imposed symmetry.

using MPSKitTaSK
using MPSKit
using MPSKitModels
using TensorKit
using LinearAlgebra

const SYMMETRIES = (Trivial, U1Irrep, SU2Irrep)

# maximal virtual spaces large enough to represent the L = 6 spin-1/2 ground state exactly
finite_max_virtualspace(::Type{Trivial}) = ℂ^8
finite_max_virtualspace(::Type{U1Irrep}) =
    U1Space(0 => 4, 1 // 2 => 4, -1 // 2 => 4, 1 => 3, -1 => 3, 3 // 2 => 1, -3 // 2 => 1)
finite_max_virtualspace(::Type{SU2Irrep}) = SU2Space(0 => 4, 1 // 2 => 4, 1 => 3, 3 // 2 => 1)

"Exact ground state of the spin-1/2 Heisenberg chain of length `L = 6` with the given symmetry."
function finite_heisenberg(symmetry; L = 6)
    H = heisenberg_XXX(ComplexF64, symmetry, FiniteChain(L); spin = 1 // 2)
    ψ0 = FiniteMPS(L, physicalspace(H, 1), finite_max_virtualspace(symmetry))
    alg = DMRG2(; trunc = truncrank(64), maxiter = 20, tol = 1.0e-10, verbosity = 0)
    gs, envs, = find_groundstate(ψ0, H, alg)
    return H, gs, envs
end

"The spin probe of the given symmetry and its multiplicity: `S_z` for abelian symmetries, `𝐒` for SU(2)."
spin_probe(::Type{SU2Irrep}; spin) = spin_vector_operator(ComplexF64, SU2Irrep; spin), 3
spin_probe(symmetry; spin) = S_z(ComplexF64, symmetry; spin), 1

"The SU(2)-invariant dimer operator `𝐒ᵢ ⋅ 𝐒ᵢ₊₁`."
dimer_probe(::Type{SU2Irrep}; spin) = spin_exchange_operator(ComplexF64, SU2Irrep; spin)
dimer_probe(symmetry; spin) = S_exchange(ComplexF64, symmetry; spin)

"Lanczos spectral moments `∑ₙ wₙ Eₙᵖ` for `p = 0, …, pmax`."
pole_moments(res::TaSKResult, pmax = 3) = [sum(res.weights .* res.energies .^ p) for p in 0:pmax]

"Overlap and distance of a TaSK and a TDVP time step of the quasiparticle `qp`."
function tdvp_agreement(qp, H, alg; dt = 0.05)
    qp_tdvp, = MPSKit.timestep(qp, H, 0.0, dt, TDVP())
    qp_task, = MPSKit.timestep(qp, H, 0.0, dt, alg)
    overlap = abs(dot(qp_tdvp, qp_task)) / (norm(qp_tdvp) * norm(qp_task))
    return overlap, norm(qp_tdvp - qp_task)
end
