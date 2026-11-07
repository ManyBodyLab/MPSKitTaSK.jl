"""
    TaSK(; krylovdim = 30, tol = MPSKit.Defaults.tol, verbosity = MPSKit.Defaults.verbosity, orth = KrylovKit.ModifiedGramSchmidt2(), kwargs...)

Tangent-space Krylov (TaSK) algorithm: a Lanczos recursion of the excitation Hamiltonian
within the quasiparticle tangent space of a ground state (arXiv:2510.07279).

`verbosity` follows the MPSKit levels (`0`: none, `1`: warnings, `2`: start/finish,
`3`: every Lanczos step), reporting the lowest Ritz value (`obj`, the lowest excitation energy resolved so far)
and its residual norm (`err`).
"""
@kwdef struct TaSK{O, E, B, F} <: MPSKit.Algorithm
    krylovdim::Int = 30
    tol::Float64 = MPSKit.Defaults.tol
    verbosity::Int = MPSKit.Defaults.verbosity
    orth::O = KrylovKit.ModifiedGramSchmidt2()
    alg_environments::E = MPSKit.Defaults.alg_environments(; dynamic_tols = false)
    backend::B = MPSKit.Defaults.backend()
    finalize::F = MPSKit.Defaults._finalize
end

"""
    TaSKResult

Result of a TaSK Lanczos run: poles `energies` with spectral `weights` (summing to `b0^2`),
the Krylov tridiagonal matrix `T_kr`, the norm `b0` of the projected initial state,
the ground-state expectation value `expval`, the initial vector `phi0` and the Krylov `basis`.
"""
struct TaSKResult{T <: Real, C <: Number, Q, M <: SymTridiagonal{T}}
    energies::Vector{T}
    weights::Vector{T}
    T_kr::M
    b0::T
    expval::C
    phi0::Q
    basis::Vector{Q}
end

"""
    task_lanczos(H, phi0::MPSKit.QP, alg::TaSK = TaSK(), lenvs = nothing, renvs = nothing; kwargs...) -> TaSKResult

Run the TaSK Lanczos recursion of the effective excitation Hamiltonian of `H` starting from the
tangent vector `phi0`.
"""
function task_lanczos(
        H, phi0::MPSKit.QP, alg::TaSK = TaSK(),
        lenvs = nothing,
        renvs = nothing;
        gs_envs = nothing,
        envs = gs_envs,
        b0::Real = 1.0,
        expval::Number = 0.0,
        store_basis::Bool = true
    )
    if isnothing(lenvs)
        lenvs = isnothing(envs) ? MPSKit.environments(phi0.left_gs, H, phi0.left_gs) : envs
    end
    if isnothing(renvs)
        renvs = !MPSKit.istopological(phi0) ? lenvs : MPSKit.environments(phi0.right_gs, H, phi0.right_gs)
    end
    TR = real(scalartype(phi0))
    TC = complex(scalartype(phi0))
    n0 = norm(phi0)
    if n0 < alg.tol
        return TaSKResult(TR[], TR[], SymTridiagonal(TR[], TR[]), TR(b0 * n0), TC(expval), phi0, store_basis ? [phi0] : typeof(phi0)[])
    end

    phi_norm = copy(phi0)
    rmul!(phi_norm, 1.0 / n0)
    total_b0 = TR(b0 * n0)

    E = MPSKit.effective_excitation_renormalization_energy(H, phi_norm, lenvs, renvs)
    H_eff = MPSKit.EffectiveExcitationHamiltonian(H, lenvs, renvs, E)

    iter = KrylovKit.LanczosIterator(x -> H_eff(x, alg.alg_environments; backend = alg.backend), phi_norm, alg.orth)
    log = IterLog("TaSK")
    fact = with_verbosity(; alg.verbosity) do
        fact = KrylovKit.initialize(iter)
        @log_initialization loginit!(log, _lowest_ritz(fact)...)
        while length(fact) < alg.krylovdim && KrylovKit.normres(fact) >= alg.tol
            fact = KrylovKit.expand!(iter, fact)
            @log_iteration logiter!(log, length(fact), _lowest_ritz(fact)...)
        end
        @log_convergence logfinish!(log, length(fact), _lowest_ritz(fact)...)
        return fact
    end

    αs = copy(fact.αs)
    βs = copy(fact.βs)
    if length(βs) == length(αs)
        pop!(βs)
    end

    T_kr = SymTridiagonal(αs, βs)
    F = eigen(T_kr)
    energies = F.values
    weights = (total_b0^2) .* abs2.(F.vectors[1, :])

    basis = store_basis ? [fact.V[i] for i in 1:length(fact)] : typeof(phi0)[]
    return TaSKResult(energies, weights, T_kr, total_b0, TC(expval), fact.V[1], basis)
end

# Residual norm `βₖ |uₖ|` and value of the lowest Ritz pair of the Lanczos factorization,
# as reported by the iteration log
function _lowest_ritz(fact)
    F = eigen(KrylovKit.rayleighquotient(fact), 1:1)
    return Float64(KrylovKit.normres(fact) * abs(F.vectors[end, 1])), only(F.values)
end
