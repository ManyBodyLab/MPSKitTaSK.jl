using Test
include("setup_models.jl")

# 2-site leakage variance σ²⊥ = ‖P^{2⊥} H |Φ⟩‖² of a normalized tangent vector
@testset "tangent_variance_2site" begin
    @testset "exact for maximal bond spaces ($symmetry)" for symmetry in SYMMETRIES
        # the L = 6 test states have maximal bond spaces: the 2-site complement is empty
        H, gs, envs = finite_heisenberg(symmetry)
        S, = spin_probe(symmetry; spin = 1 // 2)
        qp, = project_tangent(S, gs, 2π / 3)
        @test tangent_variance_2site(H, qp) == 0
    end

    @testset "truncated chain" begin
        # SU(2) chain with bond spaces far below the maximal ones
        L = 10
        H = heisenberg_XXX(ComplexF64, SU2Irrep, FiniteChain(L); spin = 1 // 2)
        Vs = [isodd(b) ? SU2Space(1 // 2 => 2, 3 // 2 => 1) : SU2Space(0 => 2, 1 => 2) for b in 1:(L - 1)]
        gs, envs, = find_groundstate(FiniteMPS(fill(SU2Space(1 // 2 => 1), L), Vs), H, DMRG(; maxiter = 50, tol = 1.0e-10, verbosity = 0))
        for O in (spin_vector_operator(ComplexF64, SU2Irrep), spin_exchange_operator(ComplexF64, SU2Irrep))
            qp, = project_tangent(O, gs, 2π * 3 / L)
            v = tangent_variance_2site(H, qp)
            @test v > 1.0e-8                                  # a nontrivial probe leaks out of the tangent space
            @test tangent_variance_2site(H, qp; envs) ≈ v    # explicit environments give the same result
            # ‖P^{2⊥} H Φ‖² ≤ ‖H Φ‖², and ⟨Φ|H²|Φ⟩ is bounded by the squared spectral range (≲ (L J)²)
            @test v < (L * 1.0)^2
        end
    end
end
