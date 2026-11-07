using BenchmarkTools
using MPSKit, MPSKitModels, TensorKit
using MPSKitTaSK

SUITE = BenchmarkGroup()

let L = 16
    H = heisenberg_XXX(SU2Irrep, FiniteChain(L); spin = 1 // 2)
    P = SU2Space(1 // 2 => 1)
    Vs = [isodd(b) ? SU2Space(1 // 2 => 2, 3 // 2 => 1) : SU2Space(0 => 2, 1 => 2) for b in 1:(L - 1)]
    gs, envs = find_groundstate(FiniteMPS(fill(P, L), Vs), H, DMRG2(; trunc = truncrank(32), verbosity = 0))
    S = spin_vector_operator(SU2Irrep)
    alg = TaSK(; krylovdim = 30, verbosity = 0)
    SUITE["project_tangent"] = @benchmarkable project_tangent($S, $gs, π)
    SUITE["task_spectrum"] = @benchmarkable task_spectrum($H, $S, $gs, π, $alg; envs = $envs)
end
