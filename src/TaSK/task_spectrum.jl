function task_lanczos(
        O, gs::MPSKit.AbstractMPS, H::MPSKit.AbstractMPO;
        k::Union{Nothing, Real} = nothing,
        momentum::Union{Nothing, Real} = k,
        pos::Union{Nothing, Int} = nothing,
        site::Union{Nothing, Int} = pos,
        alg::TaSK = TaSK(),
        amplitudes = nothing,
        coeffs = amplitudes,
        sector = nothing,
        gs_envs = nothing,
        envs = gs_envs,
        solver = MPSKit.Defaults.linearsolver,
        store_basis::Bool = true
    )
    return task_spectrum(H, O, gs, alg; site = site, momentum = momentum, amplitudes = coeffs, sector = sector, envs = envs, solver = solver, store_basis = store_basis)
end

function task_lanczos(O, pos::Int, gs::MPSKit.AbstractMPS, H::MPSKit.AbstractMPO; kwargs...)
    return task_lanczos(O, gs, H; site = pos, kwargs...)
end

function task_lanczos(O, gs::MPSKit.AbstractMPS, H::MPSKit.AbstractMPO, k::Real; kwargs...)
    return task_lanczos(O, gs, H; k = k, kwargs...)
end

function task_lanczos(
        H, O, gs::MPSKit.AbstractMPS, alg::TaSK = TaSK();
        kwargs...
    )
    return task_spectrum(H, O, gs, alg; kwargs...)
end

"""
    task_spectrum(H, O, gs::MPSKit.AbstractMPS, alg::TaSK = TaSK(); site, momentum, kwargs...) -> TaSKResult

Project `O|gs⟩` onto the quasiparticle tangent space of the finite MPS `gs` (at `site`, or summed over the chain at
`momentum` ∈ 2π/L ℤ; see [`project_tangent`](@ref)) and run the TaSK Lanczos recursion of `H` from it.
The poles and weights of the spectral function are returned as a [`TaSKResult`](@ref).
"""
function task_spectrum(
        H, O, gs::MPSKit.AbstractMPS, alg::TaSK = TaSK();
        site::Union{Nothing, Int} = nothing,
        pos::Union{Nothing, Int} = site,
        k::Union{Nothing, Real} = nothing,
        momentum::Union{Nothing, Real} = k,
        amplitudes = nothing,
        coeffs = amplitudes,
        sector = nothing,
        gs_envs = nothing,
        envs = gs_envs,
        lenvs = isnothing(envs) ? MPSKit.environments(gs, H, gs) : envs,
        solver = MPSKit.Defaults.linearsolver,
        store_basis::Bool = true
    )
    amps = !isnothing(amplitudes) ? amplitudes : coeffs
    qp, b0, expval = project_tangent(O, gs; site = pos, k = momentum, amplitudes = amps, sector = sector, left_gs = gs, solver = solver)
    renvs = !MPSKit.istopological(qp) ? lenvs : MPSKit.environments(qp.right_gs, H, qp.right_gs)
    return task_lanczos(H, qp, alg, lenvs, renvs; b0 = b0, expval = expval, store_basis = store_basis)
end

function task_spectrum(H, O, gs::MPSKit.AbstractMPS, momentum::Real, alg::TaSK = TaSK(); kwargs...)
    return task_spectrum(H, O, gs, alg; momentum = momentum, kwargs...)
end

function task_spectrum(
        O, gs::MPSKit.AbstractMPS, H::MPSKit.AbstractMPO, omega_grid::AbstractVector{<:Real};
        k::Union{Nothing, Real} = nothing,
        momentum::Union{Nothing, Real} = k,
        pos::Union{Nothing, Int} = nothing,
        site::Union{Nothing, Int} = pos,
        alg::TaSK = TaSK(),
        broadening::Symbol = :cfe,
        eta::Real = 0.08,
        sigma::Union{Nothing, Real} = nothing,
        n_cfe::Int = 2,
        amplitudes = nothing,
        coeffs = amplitudes,
        sector = nothing,
        gs_envs = nothing,
        envs = gs_envs,
        solver = MPSKit.Defaults.linearsolver,
        store_basis::Bool = true
    )
    res = task_spectrum(H, O, gs, alg; site = site, momentum = momentum, amplitudes = coeffs, sector = sector, envs = envs, solver = solver, store_basis = store_basis)
    return _broaden_task_result(res, omega_grid; broadening = broadening, eta = eta, sigma = sigma, n_cfe = n_cfe)
end

function task_spectrum(H, O, gs::MPSKit.AbstractMPS, omega_grid::AbstractVector{<:Real}; kwargs...)
    return task_spectrum(O, gs, H, omega_grid; kwargs...)
end

function task_spectrum(O, gs::MPSKit.AbstractMPS, H::MPSKit.AbstractMPO, momentum::Real, omega_grid::AbstractVector{<:Real}; kwargs...)
    return task_spectrum(O, gs, H, omega_grid; momentum = momentum, kwargs...)
end

"""
    dynamical_structure_factor(O, gs, k_grid, omega_grid; H, alg = TaSK(), broadening = :cfe, kwargs...)

Momentum-resolved dynamical structure factor `S(k, ω)` of `O` from one TaSK run per momentum.
Returns a named tuple with `k_grid`, `k_over_pi`, `omega_grid`, `S_kw` (`ω × k`), `peak_omegas`,
`weights_sum` and the individual `results`.
"""
function dynamical_structure_factor(
        O, gs::MPSKit.AbstractMPS, k_grid::AbstractVector{<:Real}, omega_grid::AbstractVector{<:Real};
        H = nothing,
        gs_envs = nothing,
        envs = gs_envs,
        alg::TaSK = TaSK(),
        broadening::Symbol = :cfe,
        eta::Real = 0.08,
        sigma::Union{Nothing, Real} = nothing,
        n_cfe::Int = 2,
        pos::Union{Nothing, Int} = nothing,
        site::Union{Nothing, Int} = pos,
        amplitudes = nothing,
        coeffs = amplitudes,
        sector = nothing,
        solver = MPSKit.Defaults.linearsolver,
        store_basis::Bool = false
    )
    isnothing(H) && throw(ArgumentError("Hamiltonian H must be provided to dynamical_structure_factor"))
    lenvs = isnothing(envs) ? MPSKit.environments(gs, H, gs) : envs
    TR = promote_type(real(scalartype(gs)), eltype(k_grid), eltype(omega_grid))
    k_vec = collect(TR, k_grid)
    k_over_pi = k_vec ./ TR(π)
    ω_vec = collect(TR, omega_grid)
    σ_val = isnothing(sigma) ? TR(eta) : TR(sigma)
    n_k = length(k_vec)
    n_ω = length(ω_vec)

    S_kw = zeros(TR, n_ω, n_k)
    peak_omegas = zeros(TR, n_k)
    weights_sum = zeros(TR, n_k)
    results = Vector{TaSKResult}(undef, n_k)

    for (j, k) in enumerate(k_vec)
        spec = task_spectrum(
            O, gs, H, ω_vec;
            momentum = k, gs_envs = lenvs, alg = alg,
            broadening = broadening, eta = eta, sigma = sigma, n_cfe = n_cfe,
            site = site, amplitudes = coeffs,
            sector = sector, solver = solver, store_basis = store_basis
        )
        res = spec.result
        S_k = spec.S_omega
        S_kw[:, j] .= S_k
        peak_omegas[j] = if !isempty(res.weights) && maximum(res.weights) > 0.0
            if σ_val > 0.0 && length(res.energies) > 1
                σ_c = TR(0.65) * σ_val
                best_idx = 1
                best_score = -1.0
                for i in eachindex(res.energies)
                    E_i = res.energies[i]
                    s = 0.0
                    for m in i:lastindex(res.energies)
                        dE = res.energies[m] - E_i
                        if dE >= 0.0
                            s += res.weights[m] * exp(-0.5 * (dE / σ_c)^2)
                        end
                    end
                    if s > best_score
                        best_score = s
                        best_idx = i
                    end
                end
                res.energies[best_idx]
            else
                res.energies[argmax(res.weights)]
            end
        elseif !isempty(ω_vec)
            ω_vec[argmax(S_k)]
        else
            zero(TR)
        end
        weights_sum[j] = res.b0^2
        results[j] = res
    end

    return (
        k_grid = k_vec,
        k_over_pi = k_over_pi,
        omega_grid = ω_vec,
        S_kw = S_kw,
        peak_omegas = peak_omegas,
        weights_sum = weights_sum,
        results = results,
    )
end

function dynamical_structure_factor(
        H, O, ψ::MPSKit.AbstractMPS, alg::TaSK = TaSK(), envs = nothing;
        k_grid = nothing,
        omega_grid = nothing,
        gs_envs = envs,
        kwargs...
    )
    actual_envs = isnothing(envs) ? (isnothing(gs_envs) ? MPSKit.environments(ψ, H, ψ) : gs_envs) : envs
    if !isnothing(k_grid) && !isnothing(omega_grid)
        return dynamical_structure_factor(O, ψ, k_grid, omega_grid; H = H, alg = alg, envs = actual_envs, kwargs...)
    elseif !isnothing(omega_grid)
        return task_spectrum(H, O, ψ, omega_grid; alg = alg, envs = actual_envs, kwargs...)
    else
        return task_spectrum(H, O, ψ, alg; envs = actual_envs, kwargs...)
    end
end

function dynamical_structure_factor(
        H, O, ψ::MPSKit.AbstractMPS, k_grid::AbstractVector{<:Real}, omega_grid::AbstractVector{<:Real},
        alg::TaSK = TaSK();
        envs = nothing,
        gs_envs = envs,
        kwargs...
    )
    actual_envs = isnothing(envs) ? (isnothing(gs_envs) ? MPSKit.environments(ψ, H, ψ) : gs_envs) : envs
    return dynamical_structure_factor(O, ψ, k_grid, omega_grid; H = H, alg = alg, envs = actual_envs, kwargs...)
end

function dynamical_structure_factor(
        O, gs::MPSKit.AbstractMPS, H::MPSKit.AbstractMPO, k_grid::AbstractVector{<:Real}, omega_grid::AbstractVector{<:Real},
        alg::TaSK = TaSK();
        kwargs...
    )
    return dynamical_structure_factor(O, gs, k_grid, omega_grid; H = H, alg = alg, kwargs...)
end

function MPSKit.excitations(
        H, alg::TaSK, O, ψ::MPSKit.AbstractMPS, envs = MPSKit.environments(ψ, H, ψ);
        kwargs...
    )
    return task_spectrum(H, O, ψ, alg; envs = envs, kwargs...)
end

function MPSKit.excitations(
        H, alg::TaSK, ψ::MPSKit.AbstractMPS, envs::MPSKit.AbstractMPSEnvironments = MPSKit.environments(ψ, H, ψ);
        O = nothing, kwargs...
    )
    isnothing(O) && throw(ArgumentError("TaSK excitation calculation requires an operator O."))
    return task_spectrum(H, O, ψ, alg; envs = envs, kwargs...)
end

"""
    evaluate_correlator(res::TaSKResult, t; include_disconnected = false)

Real-time correlator `C(t) = ∑ₙ wₙ exp(-i Eₙ t)` (plus `|⟨O⟩|²` if `include_disconnected`).
"""
function evaluate_correlator(res::TaSKResult, t::Real; include_disconnected::Bool = false)
    TC = complex(promote_type(eltype(res.energies), eltype(res.weights), typeof(res.expval), typeof(t)))
    val = isempty(res.weights) ? zero(TC) : sum(res.weights .* cis.(-res.energies .* t))
    if include_disconnected
        val += abs2(res.expval)
    end
    return val
end

function evaluate_correlator(res::TaSKResult, t_grid::AbstractVector{<:Real}; include_disconnected::Bool = false)
    return [evaluate_correlator(res, t; include_disconnected = include_disconnected) for t in t_grid]
end

"""
    dynamical_correlator(H, O, gs, alg::TaSK = TaSK(); momentum, ω_grid, t_grid, broadening = :cfe, kwargs...)

Run TaSK and return the poles together with the broadened spectral function `S_ω` on `ω_grid`
and/or the real-time correlator `C_t` on `t_grid`.
"""
function dynamical_correlator(
        H, O, gs, alg::TaSK = TaSK();
        momentum = nothing,
        ω_grid = nothing,
        t_grid = nothing,
        broadening::Symbol = :cfe,
        n_cfe::Int = 2,
        η::Real = 0.05,
        σ::Real = 0.05,
        include_disconnected::Bool = false,
        kwargs...
    )
    res = isnothing(momentum) ? task_spectrum(H, O, gs, alg; kwargs...) : task_spectrum(H, O, gs, momentum, alg; kwargs...)

    S_ω = nothing
    if !isnothing(ω_grid)
        S_ω = if broadening == :cfe
            cfe_broadening(res.energies, res.weights, ω_grid; n_cfe = n_cfe, σ = σ, η_reg = η)
        elseif broadening == :gaussian
            gaussian_broadening(res.energies, res.weights, ω_grid; σ = σ)
        elseif broadening == :lorentzian
            lorentzian_broadening(res.energies, res.weights, ω_grid; σ = η)
        else
            throw(ArgumentError("Unknown broadening method: $broadening"))
        end
    end

    C_t = isnothing(t_grid) ? nothing : evaluate_correlator(res, t_grid; include_disconnected = include_disconnected)

    return (result = res, energies = res.energies, weights = res.weights, S_ω = S_ω, C_t = C_t)
end
