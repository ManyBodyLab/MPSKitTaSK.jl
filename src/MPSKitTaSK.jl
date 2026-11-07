"""
Tangent-space Krylov (TaSK) spectral functions of finite matrix product states, built on MPSKit.jl
(arXiv:2510.07279).
"""
module MPSKitTaSK

using LinearAlgebra
using TensorKit
using KrylovKit
using MPSKit
using VectorInterface
using MPSKit: @plansor
using MPSKit.IterativeLoggers: IterLog, loginit!, logiter!, logfinish!, with_verbosity,
    @log_initialization, @log_iteration, @log_convergence
import MPSKit.MatrixAlgebraKit as MAK

export TaSK, TaSKResult
export project_tangent, project_tangent!
export task_lanczos, task_spectrum, task_time_evolve, tdvp_correlator, tdvp_structure_factor
export dynamical_structure_factor, dynamical_correlator, evaluate_correlator
export gaussian_broadening, lorentzian_broadening, cfe_broadening, spectral_moments
export broaden, spectral_map, correlator_spectrum
export tangent_variance_2site
export LocalMPO, local_mpo, spin_vector_operator, spin_exchange_operator
export plot_spectral_function

include("TaSK/algorithm.jl")
include("TaSK/local_operators.jl")
include("TaSK/projection.jl")
include("spectral_functions/broadening.jl")
include("TaSK/task_spectrum.jl")
include("TaSK/time_evolution.jl")
include("TaSK/tangent_variance.jl")

"""
    plot_spectral_function(res_dsf; title = "Dynamical Structure Factor", clims = nothing)
    plot_spectral_function(res::TaSKResult; ω_grid = nothing, broadening = :gaussian, σ = 0.05, n_cfe = 2)
    plot_spectral_function(results::AbstractVector{<:TaSKResult}, ks = 1:length(results); ω_grid = nothing, broadening = :gaussian, σ = 0.05, n_cfe = 2, logscale = false)
    plot_spectral_function(S::AbstractMatrix, ks, ω_grid; logscale = false)

Heatmap of the dynamical structure factor returned by [`dynamical_structure_factor`](@ref),
overlaid with the peak positions; or the spectral function of a single TaSK result, or a heatmap of
several (e.g. one per momentum `ks`, in units of π, shown on `xlims = (0, 2)`). With `broadening = nothing` the bare poles are shown instead,
coloured by their weight; `logscale = true` colours by its logarithm (floored 10 decades below the maximum).
A precomputed `S[ω, k]` (e.g. from [`tdvp_structure_factor`](@ref)) is plotted as a heatmap directly.
Requires `Plots.jl` to be loaded.
"""
function plot_spectral_function end

end
