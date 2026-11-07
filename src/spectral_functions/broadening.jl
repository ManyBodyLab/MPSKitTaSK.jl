# Moment-Preserving CFE Broadening and Spectral Moments

"""
    gaussian_broadening(energies::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real}, [σ::Real]; σ::Real = 0.05) -> Vector

Compute Gaussian broadened spectral function:
S(ω) = ∑ₙ Sₙ * (1 / (sqrt(2π) * σ)) * exp(- (ω - ωₙ)^2 / (2 * σ^2))
"""
function gaussian_broadening(energies::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real}, σ_pos::Real; σ::Real = σ_pos)
    return _gaussian_broadening_impl(energies, weights, ω_grid, σ)
end

function gaussian_broadening(energies::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real}; σ::Real = 0.05)
    return _gaussian_broadening_impl(energies, weights, ω_grid, σ)
end

function _gaussian_broadening_impl(energies::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real}, σ::Real)
    TR = promote_type(eltype(energies), eltype(weights), eltype(ω_grid), typeof(σ))
    S_ω = zeros(TR, length(ω_grid))
    σ_val = TR(σ)
    norm_factor = TR(1.0 / (sqrt(2.0 * π) * σ_val))
    inv_2_sigma2 = TR(0.5 / (σ_val^2))
    for (i, ω) in enumerate(ω_grid)
        val = zero(TR)
        for (E, w) in zip(energies, weights)
            diff = TR(ω - E)
            val += TR(w) * exp(-diff * diff * inv_2_sigma2)
        end
        S_ω[i] = val * norm_factor
    end
    return S_ω
end

"""
    lorentzian_broadening(energies::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real}, [σ::Real]; σ::Real = 0.05) -> Vector

Compute Lorentzian broadened spectral function with half width `σ`:
S(ω) = ∑ₙ Sₙ * (σ / π) / ((ω - ωₙ)^2 + σ^2)
"""
function lorentzian_broadening(energies::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real}, σ_pos::Real; σ::Real = σ_pos)
    return _lorentzian_broadening_impl(energies, weights, ω_grid, σ)
end

function lorentzian_broadening(energies::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real}; σ::Real = 0.05)
    return _lorentzian_broadening_impl(energies, weights, ω_grid, σ)
end

function _lorentzian_broadening_impl(energies::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real}, σ::Real)
    TR = promote_type(eltype(energies), eltype(weights), eltype(ω_grid), typeof(σ))
    S_ω = zeros(TR, length(ω_grid))
    inv_pi = TR(1.0 / π)
    σ_val = TR(σ)
    σ2 = σ_val^2
    for (i, ω) in enumerate(ω_grid)
        val = zero(TR)
        for (E, w) in zip(energies, weights)
            diff = TR(ω - E)
            val += TR(w) * (σ_val * inv_pi) / (diff * diff + σ2)
        end
        S_ω[i] = val
    end
    return S_ω
end

"""
    dawson(x::Real)

Dawson's integral `D₊(x) = exp(-x²) ∫₀ˣ exp(t²) dt`, accurate to machine precision: the Maclaurin series
`∑ₙ x²ⁿ⁺¹ / (n! (2n + 1))` (all terms positive, so free of cancellation) for `|x| ≤ 7` and the asymptotic
expansion `(1 / 2x) ∑ₖ (2k - 1)!! / (2x²)ᵏ` beyond.
"""
function dawson(x::Real)
    TR = typeof(float(x))
    ax = TR(abs(x))
    x2 = ax * ax
    if ax <= 7
        term = ax
        sum_val = zero(TR)
        n = 0
        while true
            contrib = term / (2n + 1)
            sum_val += contrib
            # converged once past the largest term (n ≈ x²); `≤` also terminates at x = 0
            ((n > x2 && contrib <= eps(TR) * sum_val) || n > 200) && break
            n += 1
            term *= x2 / n
        end
        return copysign(exp(-x2) * sum_val, x)
    else
        inv_2x2 = inv(2 * x2)
        term = one(TR)
        sum_val = one(TR)
        for k in 1:12
            term *= (2k - 1) * inv_2x2
            sum_val += term
        end
        return copysign(sum_val / (2ax), x)
    end
end

"""
    cfe_broadening(energies::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real}, [σ::Real]; n_cfe::Int = 2, σ::Real = 0.05, η_reg::Real = 1e-12) -> Vector

Moment-Preserving Continued Fraction Expansion (CFE) residual broadening from SM Sec. S-2 (Eqs. S22–S36) of arXiv:2510.07279v1.
"""
function cfe_broadening(energies::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real}, σ_pos::Real; n_cfe::Int = 2, σ::Real = σ_pos, η_reg::Real = 1.0e-12)
    return _cfe_broadening_impl(energies, weights, ω_grid, n_cfe, σ, η_reg)
end

function cfe_broadening(energies::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real}; n_cfe::Int = 2, σ::Real = 0.05, η_reg::Real = 1.0e-12)
    return _cfe_broadening_impl(energies, weights, ω_grid, n_cfe, σ, η_reg)
end

function _cfe_broadening_impl(energies::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real}, n_cfe::Int, σ::Real, η_reg::Real)
    TR = promote_type(eltype(energies), eltype(weights), eltype(ω_grid), typeof(σ), typeof(η_reg))
    TC = complex(TR)
    m0 = TR(sum(weights))
    if iszero(m0) || isempty(energies)
        return zeros(TR, length(ω_grid))
    end
    if n_cfe <= 0
        return _gaussian_broadening_impl(energies, weights, ω_grid, σ)
    end

    p = TR.(weights) ./ m0
    N_poles = length(energies)
    K_max = min(N_poles, 50)
    alphas = zeros(TR, K_max)
    betas = zeros(TR, K_max - 1)

    v = sqrt.(max.(p, zero(TR)))
    nrm = norm(v)
    if iszero(nrm)
        return zeros(TR, length(ω_grid))
    end
    v ./= nrm

    V_mat = zeros(TR, N_poles, K_max)
    V_mat[:, 1] .= v
    E_vec = TR.(energies)

    for k in 1:K_max
        vk = @view V_mat[:, k]
        w = E_vec .* vk
        # Pass 1: Full Gram-Schmidt
        alpha = zero(TR)
        for j in 1:k
            vj = @view V_mat[:, j]
            c = dot(vj, w)
            if j == k
                alpha = c
            end
            w .-= c .* vj
        end
        # Pass 2: Reorthogonalization
        for j in 1:k
            vj = @view V_mat[:, j]
            c2 = dot(vj, w)
            if j == k
                alpha += c2
            end
            w .-= c2 .* vj
        end
        alphas[k] = alpha
        if k < K_max
            beta = norm(w)
            betas[k] = beta
            if beta < 1.0e-14
                K_max = k
                resize!(alphas, K_max)
                resize!(betas, K_max - 1)
                break
            end
            V_mat[:, k + 1] .= w ./ beta
        end
    end

    m = min(n_cfe - 1, K_max - 2)
    if m < 0
        return _gaussian_broadening_impl(energies, weights, ω_grid, σ)
    end

    start_tail = m + 2
    n_tail = K_max - start_tail + 1
    M_tail = SymTridiagonal(alphas[start_tail:K_max], betas[start_tail:(K_max - 1)])

    F = eigen(M_tail)
    ω_tilde = F.values
    U = F.vectors

    t_m1 = betas[m + 1]
    tilde_w = [t_m1^2 * (U[1, ν]^2) for ν in 1:n_tail]

    σ_val = TR(σ)
    inv_root_2_sigma = TR(1.0 / (sqrt(2.0) * σ_val))
    norm_gauss = TR(1.0 / (sqrt(2.0 * π) * σ_val))
    inv_sigma_sqrt2 = TR(sqrt(2.0) / σ_val)

    R_m = Vector{TC}(undef, length(ω_grid))
    for (i, ω) in enumerate(ω_grid)
        re_val = zero(TR)
        im_val = zero(TR)
        for ν in 1:n_tail
            Δω = TR(ω) - ω_tilde[ν]
            x = Δω * inv_root_2_sigma
            im_val -= TR(π) * tilde_w[ν] * norm_gauss * exp(-x * x)
            re_val += tilde_w[ν] * inv_sigma_sqrt2 * dawson(x)
        end
        R_m[i] = complex(re_val, im_val)
    end

    current_Σ = R_m
    η_val = TR(η_reg)
    for k_idx in m:-1:1
        eps_val = alphas[k_idx + 1]
        t2 = betas[k_idx]^2
        new_Σ = similar(current_Σ)
        for (i, ω) in enumerate(ω_grid)
            denom = complex(TR(ω), η_val) - eps_val - current_Σ[i]
            new_Σ[i] = t2 / denom
        end
        current_Σ = new_Σ
    end

    S_ω = zeros(TR, length(ω_grid))
    eps_0 = alphas[1]
    for (i, ω) in enumerate(ω_grid)
        denom_0 = complex(TR(ω), η_val) - eps_0 - current_Σ[i]
        G_0 = 1.0 / denom_0
        S_ω[i] = -(m0 / TR(π)) * imag(G_0)
    end

    return S_ω
end

"""
    spectral_moments(ω_grid::AbstractVector{<:Real}, S_ω::AbstractVector{<:Real}, max_order::Int = 4; max_moment::Int = max_order) -> Vector

Compute spectral moments μ_k = ∫ ω^k S(ω) dω for k = 0, ..., max_moment.
"""
function spectral_moments(ω_grid::AbstractVector{<:Real}, S_ω::AbstractVector{<:Real}, max_order::Int = 4; max_moment::Int = max_order)
    length(ω_grid) == length(S_ω) || throw(DimensionMismatch("ω_grid and S_ω must have equal length"))
    TR = promote_type(eltype(ω_grid), eltype(S_ω))
    if length(ω_grid) < 10
        moments = zeros(TR, max_moment + 1)
        for k in 0:max_moment
            moments[k + 1] = sum(S_ω .* (ω_grid .^ k))
        end
        return moments
    end
    moments = zeros(TR, max_moment + 1)
    n = length(ω_grid)
    for k in 0:max_moment
        val = zero(TR)
        for i in 1:(n - 1)
            h = ω_grid[i + 1] - ω_grid[i]
            f_i = (ω_grid[i]^k) * S_ω[i]
            f_ip1 = (ω_grid[i + 1]^k) * S_ω[i + 1]
            val += TR(0.5) * h * (f_i + f_ip1)
        end
        moments[k + 1] = val
    end
    return moments
end

function _broaden_task_result(
        res::TaSKResult, omega_grid::AbstractVector{<:Real};
        broadening::Symbol = :cfe,
        eta::Real = 0.08,
        sigma::Union{Nothing, Real} = nothing,
        n_cfe::Int = 2
    )
    TR = promote_type(eltype(res.energies), eltype(res.weights), eltype(omega_grid))
    ω_vec = collect(TR, omega_grid)
    σ_val = isnothing(sigma) ? TR(eta) : TR(sigma)
    S_ω = if broadening == :cfe
        cfe_broadening(res.energies, res.weights, ω_vec; n_cfe = n_cfe, σ = σ_val)
    elseif broadening == :gaussian
        gaussian_broadening(res.energies, res.weights, ω_vec; σ = σ_val)
    elseif broadening == :lorentzian
        lorentzian_broadening(res.energies, res.weights, ω_vec; σ = TR(eta))
    else
        throw(ArgumentError("Unknown broadening method: $broadening"))
    end
    return (
        omega_grid = ω_vec,
        S_omega = S_ω,
        S_ω = S_ω,
        energies = res.energies,
        weights = res.weights,
        b0 = res.b0,
        expval = res.expval,
        result = res,
    )
end

"""
    broaden(res::TaSKResult, ω_grid; broadening = :gaussian, σ = 0.05, n_cfe = 2, η_reg = 1e-12) -> Vector

Broadened spectral function of the TaSK poles on `ω_grid`, with `broadening ∈ (:gaussian, :lorentzian, :cfe)`.
"""
function broaden(
        res::TaSKResult, ω_grid::AbstractVector{<:Real};
        broadening::Symbol = :gaussian, σ::Real = 0.05, n_cfe::Int = 2, η_reg::Real = 1.0e-12
    )
    return if broadening == :gaussian
        gaussian_broadening(res.energies, res.weights, ω_grid; σ)
    elseif broadening == :lorentzian
        lorentzian_broadening(res.energies, res.weights, ω_grid; σ)
    elseif broadening == :cfe
        cfe_broadening(res.energies, res.weights, ω_grid; n_cfe, σ, η_reg)
    else
        throw(ArgumentError("Unknown broadening method: $broadening"))
    end
end

"""
    spectral_map(results::AbstractVector{<:TaSKResult}, ω_grid; kwargs...) -> Matrix

Broadened spectral functions of several TaSK results (e.g. one per momentum) as a `length(ω_grid) × length(results)`
matrix. Keyword arguments are passed to [`broaden`](@ref).
"""
function spectral_map(results::AbstractVector{<:TaSKResult}, ω_grid::AbstractVector{<:Real}; kwargs...)
    return reduce(hcat, [broaden(res, ω_grid; kwargs...) for res in results])
end

"Default frequency grid covering all poles of `results` with a margin of a few `σ`."
function default_omega_grid(results::AbstractVector{<:TaSKResult}; σ::Real = 0.05, length::Int = 400)
    Es = reduce(vcat, [res.energies for res in results])
    isempty(Es) && return range(0.0, 1.0; length)
    return range(min(0.0, minimum(Es) - 5σ), maximum(Es) + 5σ; length)
end

"""
    correlator_spectrum(times, C, ω_grid; σ = 0.05) -> Vector

Spectral function from a real-time correlator `C(t) = ∑ₙ wₙ exp(-i Eₙ t)` sampled at `times` (starting at `0`):
`S(ω) = (1/π) Re ∫₀ᵀ C(t) exp(i ω t) exp(-σ² t² / 2) dt` (trapezoidal rule). For `σ T ≫ 1` this equals the
Gaussian broadening of the poles with width `σ`, cf. [`gaussian_broadening`](@ref).
"""
function correlator_spectrum(
        times::AbstractVector{<:Real}, C::AbstractVector{<:Number}, ω_grid::AbstractVector{<:Real};
        σ::Real = 0.05
    )
    length(times) == length(C) || throw(DimensionMismatch("times and C must have equal length"))
    iszero(first(times)) || throw(ArgumentError("the correlator must start at t = 0"))
    TR = promote_type(eltype(times), real(eltype(C)), eltype(ω_grid), typeof(σ))
    window = [exp(-(σ * t)^2 / 2) for t in times]
    S_ω = zeros(TR, length(ω_grid))
    for (i, ω) in enumerate(ω_grid)
        f = [real(C[j] * cis(ω * t)) * window[j] for (j, t) in enumerate(times)]
        S_ω[i] = sum((times[j + 1] - times[j]) * (f[j] + f[j + 1]) / 2 for j in 1:(length(times) - 1)) / TR(π)
    end
    return S_ω
end
