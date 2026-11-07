using Test
include("setup_models.jl")

# On exact ground states, the TaSK spectra are independent of the imposed symmetry.
# The SU(2) spin vector 𝐒 carries the weight of all three components: ∑ₐ Sᵃᵃ = 3 Sᶻᶻ.
@testset "TaSK spectra agree across symmetries" begin
    alg = TaSK(; krylovdim = 30, verbosity = 0)
    ω_grid = range(-0.5, 5.0; length = 200)
    spectrum(res, mult) = broaden(res, ω_grid; broadening = :gaussian, σ = 0.1) ./ mult

    @testset "finite Heisenberg" begin
        models = Dict(symmetry => finite_heisenberg(symmetry) for symmetry in SYMMETRIES)
        energies = [real(expectation_value(models[s][2], models[s][1])) for s in SYMMETRIES]
        @test all(≈(energies[1]; atol = 1.0e-8), energies)

        for probe in (:spin, :dimer), setting in ((; site = 3), (; momentum = 2π / 3), (; momentum = π))
            spectra = map(SYMMETRIES) do symmetry
                H, gs, envs = models[symmetry]
                O, mult = probe == :spin ? spin_probe(symmetry; spin = 1 // 2) : (dimer_probe(symmetry; spin = 1 // 2), 1)
                res = task_spectrum(H, O, gs, alg; envs, setting...)
                return pole_moments(res) ./ mult, spectrum(res, mult)
            end
            for (moments, S_ω) in spectra[2:end]
                @test moments ≈ spectra[1][1] rtol = 1.0e-6
                @test S_ω ≈ spectra[1][2] atol = 1.0e-6
            end
        end
    end
end
