"""
    task_time_evolve(res::TaSKResult, t::Real; scale_by_b0 = false)

Time-evolve the initial tangent vector within the stored Krylov basis: `exp(-i H t) |phi0⟩`.
"""
function task_time_evolve(res::TaSKResult, t::Real; scale_by_b0::Bool = false)
    isempty(res.basis) && throw(ArgumentError("TaSKResult basis was not stored (store_basis = false)"))
    if isempty(res.energies)
        phi_t = copy(res.basis[1])
        rmul!(phi_t, scale_by_b0 ? res.b0 : 0.0)
        return phi_t
    end
    m = length(res.basis)
    F = eigen(res.T_kr)
    U = F.vectors
    vals = F.values
    prefactor = scale_by_b0 ? res.b0 : 1.0
    TC = complex(promote_type(eltype(res.energies), eltype(res.weights), scalartype(res.basis[1]), typeof(t)))
    c = zeros(TC, m)
    exp_factors = cis.(-vals .* t)
    for j in 1:m
        val_j = zero(TC)
        for i in 1:m
            val_j += U[j, i] * exp_factors[i] * conj(U[1, i])
        end
        c[j] = prefactor * val_j
    end

    phi_t = zerovector(res.basis[1])
    for j in 1:m
        if abs(c[j]) > 1.0e-15
            phi_t = add!!(phi_t, res.basis[j], c[j])
        end
    end
    return phi_t
end

function task_time_evolve(
        H, phi0::MPSKit.QP, t::Real, alg::TaSK = TaSK();
        gs_envs = nothing,
        envs = gs_envs,
        lenvs = isnothing(envs) ? MPSKit.environments(phi0.left_gs, H, phi0.left_gs) : envs,
        renvs = !MPSKit.istopological(phi0) ? lenvs : MPSKit.environments(phi0.right_gs, H, phi0.right_gs),
        scale_by_b0::Bool = true
    )
    res = task_lanczos(H, phi0, alg, lenvs, renvs; b0 = 1.0, store_basis = true)
    return task_time_evolve(res, t; scale_by_b0 = scale_by_b0)
end

function MPSKit.timestep(
        phi::MPSKit.QP, H, t, dt, alg::TaSK, envs = nothing;
        gs_envs = nothing,
        lenvs = !isnothing(envs) ? envs : (!isnothing(gs_envs) ? gs_envs : MPSKit.environments(phi.left_gs, H, phi.left_gs)),
        renvs = !MPSKit.istopological(phi) ? lenvs : MPSKit.environments(phi.right_gs, H, phi.right_gs)
    )
    phi_next = task_time_evolve(H, phi, dt, alg; lenvs = lenvs, renvs = renvs, scale_by_b0 = true)
    return phi_next, lenvs
end

# NOTE: type piracy (MPSKit function and types); this TDVP step of a quasiparticle state
# serves as the reference for `timestep(…, ::TaSK)` and belongs upstream in MPSKit.
function MPSKit.timestep(
        phi::MPSKit.QP, H, t::Number, dt::Number, alg::MPSKit.TDVP, envs = nothing;
        gs_envs = nothing,
        lenvs = !isnothing(envs) ? envs : (!isnothing(gs_envs) ? gs_envs : MPSKit.environments(phi.left_gs, H, phi.left_gs)),
        renvs = !MPSKit.istopological(phi) ? lenvs : MPSKit.environments(phi.right_gs, H, phi.right_gs)
    )
    E = MPSKit.effective_excitation_renormalization_energy(H, phi, lenvs, renvs)
    H_eff = MPSKit.EffectiveExcitationHamiltonian(H, lenvs, renvs, E)
    phi_next, _ = KrylovKit.exponentiate(x -> H_eff(x, MPSKit.Defaults.alg_environments(); backend = alg.backend), -im * dt, phi, alg.integrator)
    return phi_next, lenvs
end

"""
    tdvp_correlator(H, qp::MPSKit.QP, times; alg = TDVP(), envs = nothing) -> Vector

Real-time correlator `C(t) = ⟨qp| exp(-i H t) |qp⟩` at the increasing `times`, obtained by evolving
`qp` with the TDVP integrator within the quasiparticle tangent space. This is the time-domain reference
for [`evaluate_correlator`](@ref) of a TaSK result (for a normalized `qp`, multiply by `b0^2`).
"""
function tdvp_correlator(
        H, qp::MPSKit.QP, times::AbstractVector{<:Real};
        alg::MPSKit.TDVP = MPSKit.TDVP(),
        gs_envs = nothing,
        envs = gs_envs,
        lenvs = isnothing(envs) ? MPSKit.environments(qp.left_gs, H, qp.left_gs) : envs,
        renvs = !MPSKit.istopological(qp) ? lenvs : MPSKit.environments(qp.right_gs, H, qp.right_gs)
    )
    E = MPSKit.effective_excitation_renormalization_energy(H, qp, lenvs, renvs)
    H_eff = MPSKit.EffectiveExcitationHamiltonian(H, lenvs, renvs, E)
    f = x -> H_eff(x, MPSKit.Defaults.alg_environments(); backend = alg.backend)

    C = Vector{complex(scalartype(qp))}(undef, length(times))
    ϕ = copy(qp)
    t_prev = zero(eltype(times))
    for (i, t) in enumerate(times)
        if t != t_prev
            ϕ, = KrylovKit.exponentiate(f, -im * (t - t_prev), ϕ, alg.integrator)
        end
        C[i] = dot(qp, ϕ)
        t_prev = t
    end
    return C
end

"""
    tdvp_structure_factor(H, O, gs, k_grid, ω_grid; σ = 0.05, dt = 0.1, tmax = 5 / σ, alg = TDVP(), envs = nothing, kwargs...)

Momentum-resolved dynamical structure factor from TDVP time evolution of the projected states
`O_k|gs⟩` within the tangent space, Fourier transformed with a Gaussian window of width `σ`
(see [`correlator_spectrum`](@ref)). This is directly comparable to the Gaussian broadening of the
TaSK poles from [`dynamical_structure_factor`](@ref). Remaining keyword arguments are passed to
[`project_tangent`](@ref). Returns a named tuple with `k_grid`, `k_over_pi`, `omega_grid`, `times`,
`S_kw` (`ω × k`) and the `correlators` `C_k(t)`.
"""
function tdvp_structure_factor(
        H, O, gs::MPSKit.AbstractMPS, k_grid::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real};
        σ::Real = 0.05, dt::Real = 0.1, tmax::Real = 5 / σ,
        alg::MPSKit.TDVP = MPSKit.TDVP(),
        gs_envs = nothing,
        envs = gs_envs,
        kwargs...
    )
    lenvs = isnothing(envs) ? MPSKit.environments(gs, H, gs) : envs
    TR = promote_type(real(scalartype(gs)), eltype(k_grid), eltype(ω_grid))
    times = collect(TR, range(0, tmax; step = dt))
    S_kw = zeros(TR, length(ω_grid), length(k_grid))
    correlators = Vector{Vector{complex(TR)}}(undef, length(k_grid))
    for (j, k) in enumerate(k_grid)
        qp, b0, = project_tangent(O, gs, k; kwargs...)
        C = if b0 > sqrt(eps(TR))
            renvs = !MPSKit.istopological(qp) ? lenvs : MPSKit.environments(qp.right_gs, H, qp.right_gs)
            b0^2 .* tdvp_correlator(H, qp, times; alg, lenvs, renvs)
        else
            zeros(complex(TR), length(times))
        end
        correlators[j] = C
        S_kw[:, j] .= correlator_spectrum(times, C, ω_grid; σ)
    end
    return (
        k_grid = collect(TR, k_grid),
        k_over_pi = collect(TR, k_grid) ./ TR(π),
        omega_grid = collect(TR, ω_grid),
        times = times,
        S_kw = S_kw,
        correlators = correlators,
    )
end
