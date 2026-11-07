using Test
using MPSKitTaSK
using MPSKitTaSK: dawson, TaSKResult
using LinearAlgebra: SymTridiagonal

@testset "broadening" begin
    energies = [0.5, 1.0, 2.0]
    weights = [0.2, 0.5, 0.3]
    ω = range(-10, 15; length = 20001)
    dω = step(ω)

    @test sum(gaussian_broadening(energies, weights, ω; σ = 0.1)) * dω ≈ 1 rtol = 1.0e-6
    @test sum(lorentzian_broadening(energies, weights, ω; σ = 0.05)) * dω ≈ 1 rtol = 1.0e-2
    # CFE broadening is normalized and moment preserving; on a dense band the continued fraction has no
    # (undamped, η_reg → 0) bound states outside the broadened tail that a frequency grid could miss
    band_E = collect(range(0, 3; length = 200))
    band_w = [1 + sin(3E)^2 for E in band_E]
    band_w ./= sum(band_w)
    for n_cfe in 1:3
        S_cfe = cfe_broadening(band_E, band_w, ω; σ = 0.1, n_cfe)
        @test all(≥(0), S_cfe)
        @test sum(S_cfe) * dω ≈ 1 rtol = 1.0e-8
        @test spectral_moments(collect(ω), S_cfe, 1)[2] ≈ sum(band_w .* band_E) rtol = 1.0e-8
    end

    moments = spectral_moments(collect(ω), gaussian_broadening(energies, weights, ω; σ = 0.1), 2)
    @test moments[2] ≈ sum(weights .* energies) rtol = 1.0e-6

    @test dawson(0.0) == 0
    @test dawson(1.0) ≈ 0.5380795069127684 rtol = 1.0e-10
    @test dawson(-1.0) ≈ -dawson(1.0)
    @test dawson(5.0) ≈ 0.1021340744242768 rtol = 1.0e-6
    @test dawson(10.0) ≈ 0.05025384718759853 rtol = 1.0e-8

    # the Gaussian-windowed Fourier transform of the correlator is the Gaussian broadening of its poles
    σ = 0.2
    ts = collect(range(0, 8 / σ; step = 0.02))
    C = [sum(weights .* cis.(-energies .* t)) for t in ts]
    ω_grid = collect(range(-1, 3; length = 101))
    @test correlator_spectrum(ts, C, ω_grid; σ) ≈ gaussian_broadening(energies, weights, ω_grid; σ) atol = 1.0e-6
    @test_throws DimensionMismatch correlator_spectrum(ts, C[1:(end - 1)], ω_grid)
    @test_throws ArgumentError correlator_spectrum(ts .+ 1, C, ω_grid)

    res = TaSKResult(energies, weights, SymTridiagonal(energies, [0.0, 0.0]), 1.0, 0.0im, [1.0], [[1.0]])
    @test broaden(res, ω_grid; broadening = :gaussian, σ) ≈ gaussian_broadening(energies, weights, ω_grid; σ)
    @test broaden(res, ω_grid; broadening = :lorentzian, σ) ≈ lorentzian_broadening(energies, weights, ω_grid; σ)
    @test broaden(res, ω_grid; broadening = :cfe, σ) ≈ cfe_broadening(energies, weights, ω_grid; σ)
    @test_throws ArgumentError broaden(res, ω_grid; broadening = :foo)
    @test size(spectral_map([res, res, res], ω_grid)) == (length(ω_grid), 3)
end
