# Projection of local operators and MPOs onto the quasiparticle tangent space of a finite MPS


"""
    _set_qp_from_3leg!(qp::MPSKit.LeftGaugedQP, i::Int, ϕ::MPSKit.MPSTensor)

Project a 3-leg tensor `ϕ` (V_left ⊗ V_phys ← V_right) onto `qp.VLs[i]` and store in `qp.Xs[i]`.
"""
function _set_qp_from_3leg!(qp::MPSKit.LeftGaugedQP, i::Int, ϕ::MPSKit.MPSTensor)
    @tensor X_2leg[-1; -2] := conj(qp.VLs[i][1 2; -1]) * ϕ[1 2; -2]
    u = ones(scalartype(qp.Xs[i]), space(qp.Xs[i], 2)')
    @tensor qp.Xs[i][-1; -2 -3] = X_2leg[-1; -3] * conj(u[-2])
    return qp
end

"""
    _set_qp_from_4leg!(qp::MPSKit.LeftGaugedQP, i::Int, ϕ)

Project a 4-leg tensor `ϕ` (V_left ⊗ V_phys ← V_util ⊗ V_right) onto `qp.VLs[i]` and store in `qp.Xs[i]`.
"""
function _set_qp_from_4leg!(qp::MPSKit.LeftGaugedQP, i::Int, ϕ::AbstractTensorMap)
    @tensor qp.Xs[i][-1; -2 -3] = conj(qp.VLs[i][1 2; -1]) * ϕ[1 2; -2 -3]
    return qp
end

# zero tensors `ϕ[l p; u r]` and `J[l; u r]` matching the parametrization of `qp.Xs[pos]`
_zero_util_tensor(qp, gs, pos, TC) = zeros(TC, codomain(gs.AC[pos]) ← domain(qp.Xs[pos]))

# excitation sector of the first operator in `ops`
function _ops_sector(ops)
    i = findfirst(!isnothing, ops)
    isnothing(i) && throw(ArgumentError("no operator given"))
    return _excitation_sector(local_mpo(ops[i]))
end

"""
    project_tangent!(
        qp::MPSKit.FiniteQP, ops::AbstractVector, gs::MPSKit.AbstractFiniteMPS = qp.right_gs;
        amplitudes = nothing, kwargs...
    ) -> (qp, expval)

In-place projection of `∑ℓ amplitudes[ℓ] * ops[ℓ] |gs⟩` onto the 1-site tangent space `qp::FiniteQP`,
where `ops[ℓ]` acts on the sites `ℓ, …, ℓ + N - 1`.
Arbitrary N-site operators are supported by decomposing them into local MPOs (see [`local_mpo`](@ref)).
"""
function project_tangent!(
        qp::MPSKit.FiniteQP, ops::AbstractVector, gs::MPSKit.AbstractFiniteMPS = qp.right_gs;
        amplitudes = nothing,
        kwargs...
    )
    L = length(gs)
    n_ops = length(ops)
    n_ops <= L || throw(DimensionMismatch("Expected at most $L operators for FiniteMPS of length $L, got $n_ops"))
    TC = complex(promote_type(scalartype(qp), scalartype(gs), isnothing(amplitudes) ? scalartype(qp) : eltype(amplitudes)))
    amps = isnothing(amplitudes) ? ones(TC, n_ops) : amplitudes
    length(amps) == n_ops || throw(DimensionMismatch("Expected $n_ops amplitudes, got $(length(amps))"))

    istriv = MPSKit.istrivial(qp)
    ϕ_locs = [_zero_util_tensor(qp, gs, ℓ, TC) for ℓ in 1:L]
    Js = Vector{Any}(nothing, L)
    expval = zero(TC)

    for s in 1:n_ops
        op = ops[s]
        (isnothing(op) || iszero(amps[s])) && continue
        Ws = local_mpo(op)
        e = s + length(Ws) - 1
        e <= L || throw(ArgumentError("operator at site $s acts on $(length(Ws)) sites and exceeds the chain of length $L"))
        Φs, loc_ev = _local_op_projection(Ws, gs, s, qp.left_gs)
        if istriv
            expval += TC(amps[s]) * loc_ev
        end
        for (i, Φ) in enumerate(Φs)
            ℓ = s + i - 1
            if istriv
                Φ = add!!(Φ, _with_util(gs.AC[ℓ], Φ), -loc_ev)
            end
            ϕ_locs[ℓ] = add!!(ϕ_locs[ℓ], Φ, amps[s])
            if ℓ == e
                @tensor J[-1; -2 -3] := conj(qp.left_gs.AL[ℓ][1 2; -1]) * Φ[1 2; -2 -3]
                Js[ℓ] = isnothing(Js[ℓ]) ? scale!!(J, amps[s]) : add!!(Js[ℓ], J, amps[s])
            end
        end
    end

    Z = nothing
    for ℓ in 1:L
        ϕ_tot = copy(ϕ_locs[ℓ])
        Z_next = Js[ℓ]
        if !isnothing(Z)
            @tensor ϕ_prop[-1 -2; -3 -4] := Z[-1; -3 1] * gs.AR[ℓ][1 -2; -4]
            ϕ_tot = add!!(ϕ_tot, ϕ_prop)
            @tensor Z_prop[-1; -2 -3] := conj(qp.left_gs.AL[ℓ][1 2; -1]) * ϕ_prop[1 2; -2 -3]
            Z_next = isnothing(Z_next) ? Z_prop : add!!(Z_next, Z_prop)
        end
        _set_qp_from_4leg!(qp, ℓ, ϕ_tot)
        Z = Z_next
    end

    return qp, expval
end

"""
    project_tangent(ops::AbstractVector, gs::MPSKit.AbstractFiniteMPS; amplitudes=nothing, left_gs=gs, sector=nothing, kwargs...)

Project a vector of local operators `ops` onto the finite quasiparticle tangent space of `gs`.
"""
function project_tangent(
        ops::AbstractVector, gs::MPSKit.AbstractFiniteMPS;
        amplitudes = nothing,
        left_gs::MPSKit.AbstractFiniteMPS = gs,
        sector = nothing,
        kwargs...
    )
    TC = complex(scalartype(gs))
    TR = real(scalartype(gs))
    amps_c = isnothing(amplitudes) ? ones(TC, length(ops)) : (amplitudes isa Vector{TC} ? amplitudes : collect(TC, amplitudes))
    qp = MPSKit.LeftGaugedQP(zero, left_gs, gs; sector = isnothing(sector) ? _ops_sector(ops) : sector)
    _, expval = project_tangent!(qp, ops, gs; amplitudes = amps_c)
    b0 = TR(norm(qp))
    if b0 > 0
        rmul!(qp, inv(b0))
    end
    return qp, b0, expval
end

"""
    project_tangent(op, site::Int, gs::MPSKit.AbstractFiniteMPS; amplitude=1.0, left_gs=gs, sector=nothing, kwargs...)

Project a single local operator `op` situated at `site` onto the tangent space of `gs`.
"""
function project_tangent(
        op, site::Int, gs::MPSKit.AbstractFiniteMPS;
        amplitude::Number = 1.0,
        left_gs::MPSKit.AbstractFiniteMPS = gs,
        sector = nothing,
        kwargs...
    )
    L = length(gs)
    TC = complex(scalartype(gs))
    n_ops = L - _nsites(op) + 1
    1 <= site <= n_ops || throw(BoundsError(1:n_ops, site))
    ops = Vector{Any}(nothing, n_ops)
    ops[site] = op
    amps = zeros(TC, n_ops)
    amps[site] = TC(amplitude)
    return project_tangent(ops, gs; amplitudes = amps, left_gs = left_gs, sector = sector)
end

# site-resolved projection of a single operator (or weighted collection) onto a finite MPS,
# used by the unified entry point when no momentum is given
function _project_tangent_sites(
        op, gs::MPSKit.AbstractFiniteMPS;
        site::Union{Nothing, Int} = nothing,
        amplitude::Number = 1.0,
        amplitudes = nothing,
        coeffs = amplitudes,
        left_gs::MPSKit.AbstractFiniteMPS = gs,
        sector = nothing,
        kwargs...
    )
    amps = !isnothing(amplitudes) ? amplitudes : coeffs
    if !isnothing(amps)
        TC = complex(scalartype(gs))
        amps_c = amps isa Vector{TC} ? amps : collect(TC, amps)
        ops = fill(op, length(amps_c))
        return project_tangent(ops, gs; amplitudes = amps_c, left_gs = left_gs, sector = sector)
    end
    n_ops = length(gs) - _nsites(op) + 1
    pos_val = isnothing(site) ? cld(n_ops, 2) : site
    return project_tangent(op, pos_val, gs; amplitude = amplitude, left_gs = left_gs, sector = sector)
end

"""
    project_tangent(ops::AbstractVector, gs::MPSKit.AbstractFiniteMPS, momentum::Real; amplitudes=nothing, left_gs=gs, sector=nothing, kwargs...)

Project `∑ⱼ amplitudes[j] * exp(i k j) * ops[j] |gs⟩` at momentum `k = momentum` onto the finite
quasiparticle tangent space of `gs`. `k` must be a multiple of `2π / length(gs)`.
"""
function project_tangent(
        ops::AbstractVector, gs::MPSKit.AbstractFiniteMPS, momentum::Real;
        amplitudes = nothing,
        left_gs::MPSKit.AbstractFiniteMPS = gs,
        sector = nothing,
        kwargs...
    )
    _check_finite_momentum(momentum, length(gs))
    TC = complex(scalartype(gs))
    n = length(ops)
    amps = isnothing(amplitudes) ? fill(TC(1.0 / sqrt(n)), n) : collect(TC, amplitudes)
    length(amps) == n || throw(DimensionMismatch("Expected $n amplitudes, got $(length(amps))"))
    amps_k = [amps[j] * cis(momentum * j) for j in 1:n]
    return project_tangent(ops, gs; amplitudes = amps_k, left_gs = left_gs, sector = sector)
end

"""
    project_tangent(op, gs::MPSKit.AbstractFiniteMPS, momentum::Real; site=nothing, amplitude=1.0, amplitudes=nothing, left_gs=gs, sector=nothing, kwargs...)

Project an operator `op` at momentum `momentum` onto `gs::AbstractFiniteMPS`:
without `site`, `op` is placed on all possible positions with amplitude `amplitude / √N_ops`.
"""
function project_tangent(
        op, gs::MPSKit.AbstractFiniteMPS, momentum::Real;
        pos::Union{Nothing, Int} = nothing,
        site::Union{Nothing, Int} = pos,
        amplitude::Number = 1.0,
        amplitudes = nothing,
        coeffs = amplitudes,
        left_gs::MPSKit.AbstractFiniteMPS = gs,
        sector = nothing,
        envs = nothing,
        kwargs...
    )
    TC = complex(scalartype(gs))
    amps_in = !isnothing(amplitudes) ? amplitudes : coeffs
    if !isnothing(amps_in)
        ops = fill(op, length(amps_in))
        return project_tangent(ops, gs, momentum; amplitudes = amps_in, left_gs = left_gs, sector = sector)
    end
    n_ops = length(gs) - _nsites(op) + 1
    ops = fill(op, n_ops)
    amps = zeros(TC, n_ops)
    if isnothing(site)
        fill!(amps, TC(amplitude / sqrt(n_ops)))
    else
        1 <= site <= n_ops || throw(BoundsError(1:n_ops, site))
        amps[site] = TC(amplitude)
    end
    return project_tangent(ops, gs, momentum; amplitudes = amps, left_gs = left_gs, sector = sector)
end

# an MPO on a finite chain fixes its own spatial structure: only zero momentum is meaningful
function project_tangent(
        O::Union{MPSKit.FiniteMPO, MPSKit.FiniteMPOHamiltonian}, gs::MPSKit.AbstractFiniteMPS, momentum::Real;
        kwargs...
    )
    isapprox(mod2pi(momentum + π) - π, 0; atol = 1.0e-8) ||
        throw(ArgumentError("an MPO on a finite chain can only be projected at zero momentum, got $momentum"))
    return project_tangent(O, gs; kwargs...)
end

function _check_finite_momentum(k::Real, L::Int)
    n = k * L / (2π)
    isapprox(n, round(n); atol = 1.0e-8) ||
        throw(ArgumentError("momentum $k of a finite MPS of length $L must be a multiple of 2π/$L"))
    return nothing
end

"""
    project_tangent(O::Union{MPSKit.FiniteMPO, MPSKit.FiniteMPOHamiltonian}, gs::MPSKit.AbstractFiniteMPS; left_gs=gs, sector=one(sectortype(gs)), envs=nothing, kwargs...)

Project an MPO or MPOHamiltonian onto the finite quasiparticle tangent space.
"""
function project_tangent(
        O::Union{MPSKit.FiniteMPO, MPSKit.FiniteMPOHamiltonian}, gs::MPSKit.AbstractFiniteMPS;
        left_gs::MPSKit.AbstractFiniteMPS = gs,
        sector = nothing,
        envs = nothing,
        kwargs...
    )
    TC = complex(scalartype(gs))
    TR = real(scalartype(gs))
    mpo_envs = isnothing(envs) ? MPSKit.environments(left_gs, O, gs) : envs
    qp = MPSKit.LeftGaugedQP(zero, left_gs, gs; sector = something(sector, one(sectortype(gs))))
    for ℓ in 1:length(gs)
        H_ac = MPSKit.AC_hamiltonian(ℓ, gs, O, left_gs, mpo_envs)
        ϕ = H_ac(gs.AC[ℓ])
        _set_qp_from_3leg!(qp, ℓ, ϕ)
    end
    expval = MPSKit.istrivial(qp) ? TC(MPSKit.expectation_value(gs, O, mpo_envs)) : zero(TC)
    b0 = TR(norm(qp))
    if b0 > 0
        rmul!(qp, inv(b0))
    end
    return qp, b0, expval
end

"""
    project_tangent(op, gs::MPSKit.AbstractFiniteMPS; site = nothing, momentum = nothing, kwargs...) -> (qp, b0, expval)

Project `op|gs⟩` onto the one-site quasiparticle tangent space of the finite MPS `gs` and return the
normalized tangent vector `qp`, its norm `b0` and the ground-state expectation value `expval = ⟨gs|op|gs⟩`
(subtracted from the projection; zero for charged operators).

`op` may be an N-site `AbstractTensorMap`, a [`LocalMPO`](@ref), a vector of such operators (one per site)
or a `FiniteMPO`/`FiniteMPOHamiltonian`. Without `momentum`, a single operator is placed at `site`
(default: the centre of the chain); with `momentum = 2πn/L`, it is summed over all sites with the
plane-wave amplitudes `exp(i k j) / √N`. `pos` and `k` are aliases of `site` and `momentum`.
"""
function project_tangent(
        op, gs::MPSKit.AbstractFiniteMPS;
        pos::Union{Nothing, Int} = nothing,
        site::Union{Nothing, Int} = pos,
        k::Union{Nothing, Real} = nothing,
        momentum::Union{Nothing, Real} = k,
        kwargs...
    )
    isnothing(momentum) && return _project_tangent_sites(op, gs; site, kwargs...)
    return project_tangent(op, gs, momentum; site, kwargs...)
end

project_tangent(op, pos::Int, gs::MPSKit.AbstractFiniteMPS, k::Real; kwargs...) =
    project_tangent(op, gs, k; site = pos, kwargs...)
