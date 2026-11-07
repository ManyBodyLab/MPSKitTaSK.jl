module MPSKitTaSKPlotsExt

using MPSKitTaSK
using MPSKitTaSK: TaSKResult, broaden, spectral_map, default_omega_grid
using Plots

function MPSKitTaSK.plot_spectral_function(res_dsf::NamedTuple; title = "Dynamical Structure Factor", clims = nothing, xlims = (0, 2))
    k_over_pi = res_dsf.k_over_pi
    ω_grid = res_dsf.omega_grid
    S_kw = res_dsf.S_kw

    heatmap(
        k_over_pi, ω_grid, S_kw,
        xlabel = "k / π",
        ylabel = "ω",
        title = title,
        color = :viridis,
        clims = isnothing(clims) ? (0, maximum(S_kw)) : clims,
        xlims = xlims
    )

    # Overlay the peak omegas
    return plot!(
        k_over_pi, res_dsf.peak_omegas,
        color = :white,
        linestyle = :dash,
        linewidth = 2,
        label = "Peak max"
    )
end

# a single TaSK result: broadened line plot, or the bare poles as sticks
function MPSKitTaSK.plot_spectral_function(
        res::TaSKResult;
        ω_grid = nothing, broadening::Union{Nothing, Symbol} = :gaussian, σ::Real = 0.05,
        n_cfe::Int = 2, title = "Spectral function", kwargs...
    )
    if isnothing(broadening)
        return plot(
            res.energies, res.weights;
            seriestype = :sticks, xlabel = "ω", ylabel = "weight", title, label = "poles", kwargs...
        )
    end
    ωs = isnothing(ω_grid) ? default_omega_grid([res]; σ) : ω_grid
    return plot(
        ωs, broaden(res, ωs; broadening, σ, n_cfe);
        xlabel = "ω", ylabel = "S(ω)", title, label = string(broadening), kwargs...
    )
end

# log₁₀ of non-negative values, floored `decades` below their maximum
function _log10_floor(v; decades = 10)
    m = maximum(v; init = 0.0)
    iszero(m) && return zero.(float.(v))
    return log10.(max.(v, m * 10.0^-decades))
end

# several TaSK results (e.g. one per momentum `ks`, in units of π): heatmap of the broadened spectra,
# or the bare poles coloured by their weight; `logscale` colours by log₁₀ instead
function MPSKitTaSK.plot_spectral_function(
        results::AbstractVector{<:TaSKResult}, ks::AbstractVector{<:Real} = 1:length(results);
        ω_grid = nothing, broadening::Union{Nothing, Symbol} = :gaussian, σ::Real = 0.05,
        n_cfe::Int = 2, logscale::Bool = false, xlabel = "k / π", title = "Spectral function",
        clims = nothing, xlims = (0, 2), kwargs...
    )
    length(ks) == length(results) || throw(DimensionMismatch("need one momentum per result"))
    if isnothing(broadening)
        xs = reduce(vcat, [fill(k, length(res.energies)) for (k, res) in zip(ks, results)])
        Es = reduce(vcat, [res.energies for res in results])
        ws = reduce(vcat, [res.weights for res in results])
        return scatter(
            xs, Es;
            marker_z = logscale ? _log10_floor(ws) : ws, colorbar_title = logscale ? "log₁₀ weight" : "weight",
            color = :viridis, markerstrokewidth = 0, label = "",
            xlabel, ylabel = "ω", title, xlims, kwargs...
        )
    end
    ωs = isnothing(ω_grid) ? default_omega_grid(results; σ) : ω_grid
    S = spectral_map(results, ωs; broadening, σ, n_cfe)
    return MPSKitTaSK.plot_spectral_function(S, ks, ωs; logscale, xlabel, title, clims, xlims, kwargs...)
end

# a precomputed spectral function `S[ω, k]` (e.g. from `tdvp_structure_factor`) as a heatmap
function MPSKitTaSK.plot_spectral_function(
        S::AbstractMatrix{<:Real}, ks::AbstractVector{<:Real}, ω_grid::AbstractVector{<:Real};
        logscale::Bool = false, xlabel = "k / π", title = "Spectral function",
        clims = nothing, xlims = (0, 2), kwargs...
    )
    size(S) == (length(ω_grid), length(ks)) || throw(DimensionMismatch("S must be of size (length(ω_grid), length(ks))"))
    S_plot = logscale ? _log10_floor(S) : S
    return heatmap(
        ks, ω_grid, S_plot;
        xlabel, ylabel = "ω", title, xlims, color = :viridis, colorbar_title = logscale ? "log₁₀ S(k, ω)" : "",
        clims = isnothing(clims) ? (logscale ? extrema(S_plot) : (0, maximum(S))) : clims, kwargs...
    )
end

end
