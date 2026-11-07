using Pkg; Pkg.activate(@__DIR__)
using MPSKitTaSK
using MPSKit
using MPSKitModels
using TensorKit
using LinearAlgebra
using Plots

# two-spinon continuum of the spin-1/2 Heisenberg chain (J = 1): des Cloizeaux–Pearson lower bound
# (π/2)|sin k| and upper bound π|sin(k/2)|
function add_spinon_bounds!(p; color = :red)
    ks = range(0, 2π; length = 401)
    plot!(p, ks ./ π, (π / 2) .* abs.(sin.(ks)); color, linestyle = :dash, linewidth = 1.5, label = "(π/2)|sin k|")
    plot!(p, ks ./ π, π .* abs.(sin.(ks ./ 2)); color, linestyle = :dot, linewidth = 1.5, label = "π|sin(k/2)|")
    return p
end

function main()
    println("=== TaSK vs TDVP: Finite SU(2) Heisenberg Chain ===")
    L = 16
    H = heisenberg_XXX(SU2Irrep, FiniteChain(L); spin = 1 // 2)

    # 1. Ground state via two-site DMRG
    println("Finding ground state with DMRG2...")
    P = SU2Space(1 // 2 => 1)
    # the L - 1 inner bonds alternate between half-integer (odd bonds) and integer (even bonds) spins
    V_half = SU2Space(1 // 2 => 2, 3 // 2 => 1)
    V_int = SU2Space(0 => 2, 1 => 2)
    Vs = [isodd(bond) ? V_half : V_int for bond in 1:(L - 1)]
    dmrg_alg = DMRG2(; trunc = truncrank(64), maxiter = 10, tol = 1.0e-8, verbosity = 0)
    gs, envs = find_groundstate(FiniteMPS(fill(P, L), Vs), H, dmrg_alg)

    # TaSK Algorithm
    alg = TaSK(krylovdim = 50, verbosity = 3)

    # spin vector 𝐒 (spin-1 quasiparticle sector) and SU(2)-symmetric dimer 𝐒ᵢ ⋅ 𝐒ᵢ₊₁
    S = spin_vector_operator(SU2Irrep)
    D_op = spin_exchange_operator(ComplexF64, SU2Irrep)

    # 2. Spin Tangent State & Wavefunction Time Evolution Agreement
    println("\n--- 1. Spin Quasiparticle & Time Evolution Agreement ---")
    site = cld(L, 2)
    qp_spin, b0_spin, expval_spin = project_tangent(S, site, gs)
    rmul!(qp_spin, b0_spin)

    dt = 0.05
    t_eval = dt
    # TDVP time stepping
    qp_tdvp, _ = MPSKit.timestep(qp_spin, H, 0.0, t_eval, TDVP())
    # TaSK KrylovKit-backed solver via timestep(..., alg)
    qp_task, _ = MPSKit.timestep(qp_spin, H, 0.0, t_eval, alg)

    diff_spin = norm(qp_tdvp - qp_task)
    overlap_spin = abs(dot(qp_tdvp, qp_task) / (norm(qp_tdvp) * norm(qp_task)))
    println("Spin Wavefunction Norm Diff: ", diff_spin)
    println("Spin Wavefunction Overlap:   ", overlap_spin)

    # 3. Dimer Tangent State & Wavefunction Time Evolution Agreement
    println("\n--- 2. Dimer Quasiparticle & Time Evolution Agreement ---")
    qp_dimer, b0_dimer, expval_dimer = project_tangent(D_op, site, gs)
    rmul!(qp_dimer, b0_dimer)

    qp_tdvp_d, _ = MPSKit.timestep(qp_dimer, H, 0.0, t_eval, TDVP())
    qp_task_d, _ = MPSKit.timestep(qp_dimer, H, 0.0, t_eval, alg)

    diff_dimer = norm(qp_tdvp_d - qp_task_d)
    overlap_dimer = abs(dot(qp_tdvp_d, qp_task_d) / (norm(qp_tdvp_d) * norm(qp_task_d)))
    println("Dimer Wavefunction Norm Diff:  ", diff_dimer)
    println("Dimer Wavefunction Overlap:    ", overlap_dimer)

    # 4. Dynamical Structure Factors S(k, ω): TaSK for all momenta k = 2πn/L
    println("\n--- 3. TaSK Dynamical Structure Factors for all momenta k = 2πn/L ---")
    k_grid = [2π * n / L for n in 0:(L - 1)]
    ω_grid = collect(range(0.0, 3.5; length = 300))
    σ = 0.05 # Gaussian broadening of the TaSK poles

    println("Computing Spin Structure Factor ∑ₐ Sᵃᵃ(k, ω)...")
    dsf_spin = dynamical_structure_factor(S, gs, H, k_grid, ω_grid, alg; envs, broadening = :gaussian, sigma = σ)
    for (ik, kv) in enumerate(k_grid)
        println("  Spin  k = ", round(kv / π; digits = 3), "π: Peak ω = ", round(dsf_spin.peak_omegas[ik]; digits = 4), ", Total Weight b0^2 = ", round(dsf_spin.weights_sum[ik]; digits = 4))
    end

    println("Computing Dimer Structure Factor S_dimer(k, ω)...")
    dsf_dimer = dynamical_structure_factor(D_op, gs, H, k_grid, ω_grid, alg; envs, broadening = :gaussian, sigma = σ)

    # 5. TDVP reference: evolve the projected states in time and Fourier transform them with a Gaussian
    #    window of width σ_tdvp, which reproduces the Gaussian broadening of the TaSK poles. To keep TDVP
    #    affordable it runs on a coarser momentum mesh, with a wider σ_tdvp (time window tmax = 5 / σ_tdvp)
    #    and a coarse sampling dt (accurate as long as 2π / dt exceeds the spectral range)
    println("\n--- 4. TDVP Structure Factor for comparison ---")
    σ_tdvp = 0.1
    idx = 1:2:length(k_grid)
    tdvp_spin = tdvp_structure_factor(H, S, gs, k_grid[idx], ω_grid; σ = σ_tdvp, dt = 0.5, envs)
    S_task = spectral_map(dsf_spin.results[idx], ω_grid; broadening = :gaussian, σ = σ_tdvp)
    println("Spin:  max |S_TaSK - S_TDVP| / max S_TaSK at σ = $σ_tdvp: ", maximum(abs.(S_task .- tdvp_spin.S_kw)) / maximum(S_task))
    tdvp_dimer = tdvp_structure_factor(H, D_op, gs, k_grid[idx], ω_grid; σ = σ_tdvp, dt = 0.5, envs)
    D_task = spectral_map(dsf_dimer.results[idx], ω_grid; broadening = :gaussian, σ = σ_tdvp)
    println("Dimer: max |S_TaSK - S_TDVP| / max S_TaSK at σ = $σ_tdvp: ", maximum(abs.(D_task .- tdvp_dimer.S_kw)) / maximum(D_task))

    # 6. Heatmaps: TaSK on the fine momentum mesh, TDVP on the coarse one, and the bare TaSK poles
    p1 = plot_spectral_function(dsf_spin.results, dsf_spin.k_over_pi; ω_grid, broadening = :gaussian, σ, title = "TaSK ∑ₐ Sᵃᵃ(k, ω), σ = $σ")
    p2 = plot_spectral_function(tdvp_spin.S_kw, tdvp_spin.k_over_pi, ω_grid; title = "TDVP ∑ₐ Sᵃᵃ(k, ω), σ = $σ_tdvp")
    p3 = plot_spectral_function(dsf_spin.results, dsf_spin.k_over_pi; broadening = nothing, logscale = true, title = "TaSK poles (spin)")
    p4 = plot_spectral_function(dsf_dimer.results, dsf_dimer.k_over_pi; ω_grid, broadening = :gaussian, σ, title = "TaSK S_dimer(k, ω), σ = $σ")
    p5 = plot_spectral_function(tdvp_dimer.S_kw, tdvp_dimer.k_over_pi, ω_grid; title = "TDVP S_dimer(k, ω), σ = $σ_tdvp")
    p6 = plot_spectral_function(dsf_dimer.results, dsf_dimer.k_over_pi; broadening = nothing, logscale = true, title = "TaSK poles (dimer)")
    foreach(add_spinon_bounds!, (p1, p2, p3))
    savefig(
        plot(p1, p2, p3, p4, p5, p6; layout = (2, 3), size = (1650, 850), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm),
        joinpath(@__DIR__, "heisenberg_finite.png")
    )
    return println("=== Finite Heisenberg Example Complete ===")
end

main()
