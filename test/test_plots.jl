using Test
using Plots: Plots
include("setup_models.jl")

@testset "Plots extension" begin
    H, gs, envs = finite_heisenberg(SU2Irrep)
    S, = spin_probe(SU2Irrep; spin = 1 // 2)
    alg = TaSK(; krylovdim = 10, verbosity = 0)
    ks = [2π * n / length(gs) for n in 0:(length(gs) - 1)]
    ω_grid = collect(range(0.0, 3.0; length = 50))
    dsf = dynamical_structure_factor(S, gs, H, ks, ω_grid, alg; envs, broadening = :gaussian, sigma = 0.1)
    res = dsf.results[3]

    @test plot_spectral_function(dsf) isa Plots.Plot
    @test plot_spectral_function(res) isa Plots.Plot
    @test plot_spectral_function(res; broadening = nothing) isa Plots.Plot
    @test plot_spectral_function(dsf.results, dsf.k_over_pi; ω_grid) isa Plots.Plot
    @test plot_spectral_function(dsf.results, dsf.k_over_pi; broadening = nothing, logscale = true) isa Plots.Plot
    @test plot_spectral_function(dsf.S_kw, dsf.k_over_pi, ω_grid; logscale = true) isa Plots.Plot
    @test_throws DimensionMismatch plot_spectral_function(dsf.results, [0.0])
end
