using Test
include("setup_models.jl")

@testset "TaSK time evolution" begin
    H, gs, envs = finite_heisenberg(SU2Irrep)
    L = length(gs)
    S, = spin_probe(SU2Irrep; spin = 1 // 2)
    alg = TaSK(; krylovdim = 40, verbosity = 0)
    k = 2π / L
    res = task_spectrum(H, S, gs, k, alg; envs)
    qp, b0, = project_tangent(S, gs, k)

    @testset "Krylov propagation" begin
        @test norm(task_time_evolve(res, 0.0; scale_by_b0 = true) - b0 * res.phi0) < 1.0e-10
        for t in (0.3, 1.0, 3.0)
            ϕt = task_time_evolve(res, t)
            @test norm(ϕt) ≈ 1 atol = 1.0e-10
            # ⟨φ₀|exp(-iHt)|φ₀⟩ b0² is the correlator of the poles
            @test dot(res.phi0, ϕt) * b0^2 ≈ evaluate_correlator(res, t) atol = 1.0e-10
        end
        @test_throws ArgumentError task_time_evolve(task_spectrum(H, S, gs, k, alg; envs, store_basis = false), 1.0)
    end

    D = dimer_probe(SU2Irrep; spin = 1 // 2)
    probes = ("spin" => S, "dimer" => D)

    @testset "$name correlator agrees with TDVP" for (name, O) in probes
        resO = task_spectrum(H, O, gs, k, alg; envs)
        qpO, b0O, = project_tangent(O, gs, k)
        ts = collect(range(0.0, 4.0; length = 9))
        C_tdvp = b0O^2 .* tdvp_correlator(H, qpO, ts; envs)
        @test C_tdvp ≈ evaluate_correlator(resO, ts) atol = 1.0e-6
    end

    @testset "$name structure factor agrees with TDVP" for (name, O) in probes
        σ = 0.4
        ks = [2π * n / L for n in 0:(L - 1)]
        ω_grid = collect(range(-0.5, 5.0; length = 80))
        tdvp = tdvp_structure_factor(H, O, gs, ks, ω_grid; σ, dt = 0.1, envs)
        dsf = dynamical_structure_factor(O, gs, H, ks, ω_grid, alg; envs)
        S_task = spectral_map(dsf.results, ω_grid; broadening = :gaussian, σ)
        @test size(tdvp.S_kw) == size(S_task)
        @test maximum(abs.(tdvp.S_kw .- S_task)) < 1.0e-3 * maximum(S_task)
    end
end
