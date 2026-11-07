using Test
include("setup_models.jl")

# TaSK time evolution within the Krylov space must reproduce the TDVP step of the quasiparticle
@testset "TaSK agrees with TDVP" begin
    alg = TaSK(; krylovdim = 20, verbosity = 0)

    @testset "finite Heisenberg, $symmetry" for symmetry in SYMMETRIES
        H, gs, = finite_heisenberg(symmetry)
        S, = spin_probe(symmetry; spin = 1 // 2)
        D = dimer_probe(symmetry; spin = 1 // 2)
        for (qp, b0, _) in (project_tangent(S, 3, gs), project_tangent(D, 3, gs), project_tangent(S, gs, 2π / 3))
            rmul!(qp, b0)
            overlap, dist = tdvp_agreement(qp, H, alg)
            @test overlap ≈ 1 atol = 1.0e-10
            @test dist < 1.0e-8
        end
    end
end
