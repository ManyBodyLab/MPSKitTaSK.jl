using Test
include("setup_models.jl")

@testset "TaSK spectral functions" begin
    H, gs, envs = finite_heisenberg(SU2Irrep)
    L = length(gs)
    S, = spin_probe(SU2Irrep; spin = 1 // 2)
    D = dimer_probe(SU2Irrep; spin = 1 // 2)
    alg = TaSK(; krylovdim = 30, verbosity = 0)
    ω_grid = collect(range(-0.5, 4.0; length = 150))
    σ = 0.1
    k = 2π * 2 / L

    res = task_spectrum(H, S, gs, k, alg; envs)
    @testset "sum rule and normalization" begin
        @test sum(res.weights) ≈ res.b0^2
        @test res.b0 ≈ project_tangent(S, gs, k)[2]
        @test all(≥(-1.0e-12), res.weights)
        @test evaluate_correlator(res, 0.0) ≈ res.b0^2
        @test evaluate_correlator(res, [0.0, 1.0])[1] ≈ res.b0^2
        # the lowest pole lies above the ground state (excitation energies are measured from E₀)
        @test minimum(res.energies) > 0
    end

    @testset "zero weight of the total spin on the singlet" begin
        res0 = task_spectrum(H, S, gs, 0.0, alg; envs)
        @test sum(res0.weights) < 1.0e-12
    end

    @testset "disconnected part" begin
        resD = task_spectrum(H, D, gs, alg; site = 3, envs)
        @test abs(resD.expval) > 0.1
        @test evaluate_correlator(resD, 0.0; include_disconnected = true) ≈ resD.b0^2 + abs2(resD.expval)
    end

    @testset "equivalent entry points" begin
        @test task_spectrum(H, S, gs, alg; momentum = k, envs).energies ≈ res.energies
        @test task_lanczos(S, gs, H, k; alg, envs).energies ≈ res.energies
        @test MPSKit.excitations(H, alg, S, gs, envs; momentum = k).energies ≈ res.energies
        spec = task_spectrum(H, S, gs, ω_grid; momentum = k, alg, envs, broadening = :gaussian, sigma = σ)
        @test spec.S_ω ≈ broaden(res, ω_grid; broadening = :gaussian, σ)
        dc = dynamical_correlator(H, S, gs, alg; momentum = k, envs, ω_grid, t_grid = [0.0, 0.5], broadening = :gaussian, σ)
        @test dc.S_ω ≈ broaden(res, ω_grid; broadening = :gaussian, σ)
        @test dc.C_t ≈ evaluate_correlator(res, [0.0, 0.5])
    end

    @testset "dynamical structure factor" begin
        ks = [2π * n / L for n in 0:(L - 1)]
        dsf = dynamical_structure_factor(S, gs, H, ks, ω_grid, alg; envs, broadening = :gaussian, sigma = σ)
        @test size(dsf.S_kw) == (length(ω_grid), length(ks))
        @test dsf.k_over_pi ≈ ks ./ π
        @test dsf.weights_sum ≈ [project_tangent(S, gs, q)[2]^2 for q in ks] atol = 1.0e-10
        for j in eachindex(ks)
            @test dsf.S_kw[:, j] ≈ broaden(dsf.results[j], ω_grid; broadening = :gaussian, σ) atol = 1.0e-10
        end
        @test spectral_map(dsf.results, ω_grid; broadening = :gaussian, σ) ≈ dsf.S_kw atol = 1.0e-10
        # S(k, ω) = S(2π - k, ω) for a reflection-symmetric chain
        @test dsf.S_kw[:, 2:end] ≈ reverse(dsf.S_kw[:, 2:end]; dims = 2) atol = 1.0e-6
    end
end
