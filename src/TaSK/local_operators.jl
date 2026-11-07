# Local N-site operators as local MPOs, following `MPSKit.correlator`

"""
    LocalMPO(Ws::Vector{<:MPOTensor})

An operator acting on `length(Ws)` consecutive sites, given as a local MPO `[W_1, …, W_N]`.
The left virtual leg of `W_1` must be trivial; the right virtual leg of `W_N` is the auxiliary leg
of the operator, which carries the excitation sector of charged operators.
"""
struct LocalMPO{W <: MPSKit.MPOTensor}
    Ws::Vector{W}
end
Base.length(O::LocalMPO) = length(O.Ws)

"""
    local_mpo(O) -> Vector{<:MPOTensor}

Decompose an N-site operator `O::AbstractTensorMap{T, S, N, N}` into a local MPO `[W_1, …, W_N]`
with trivial outer virtual legs, using `MPSKit.decompose_localmpo(MPSKit.add_util_leg(O))`.
A [`LocalMPO`](@ref) is returned as is.

The left virtual leg of `W_1` must be trivial. The right virtual leg of `W_N` is the auxiliary leg of
the operator: trivial for symmetric operators, and carrying the excitation sector for charged ones
(e.g. the spin-1 leg of [`spin_vector_operator`](@ref)).
"""
local_mpo(O::AbstractTensorMap{<:Any, S, N, N}) where {S, N} =
    MPSKit.decompose_localmpo(MPSKit.add_util_leg(O))
local_mpo(O::LocalMPO) = O.Ws

"""
    spin_vector_operator([elt = ComplexF64], [SU2Irrep]; spin = 1 // 2) -> LocalMPO

The spin vector operator `𝐒` as a one-site local MPO whose right virtual leg carries spin 1,
obtained from the first factor of the local MPO of `𝐒ᵢ ⋅ 𝐒ⱼ` and normalized to `𝐒 ⋅ 𝐒 = s(s+1)`.
For an SU(2)-symmetric ground state, its spectral function is `∑ₐ Sᵃᵃ(k, ω) = 3 Sᶻᶻ(k, ω)`.
"""
function spin_vector_operator(elt::Type{<:Number}, ::Type{SU2Irrep} = SU2Irrep; spin = 1 // 2)
    W = first(local_mpo(spin_exchange_operator(elt, SU2Irrep; spin)))
    @tensor SS[-1; -2] := W[1 -1; 2 3] * conj(W[1 -2; 2 3])
    SS_val = real(tr(SS)) / dim(space(SS, 1))
    return LocalMPO([scale!(W, sqrt(spin * (spin + 1) / SS_val))])
end
spin_vector_operator(::Type{SU2Irrep} = SU2Irrep; kwargs...) =
    spin_vector_operator(ComplexF64, SU2Irrep; kwargs...)

"""
    spin_exchange_operator([elt = ComplexF64], [SU2Irrep]; spin = 1 // 2) -> TensorMap

The SU(2)-invariant two-site Heisenberg exchange `𝐒ᵢ ⋅ 𝐒ⱼ = (S_tot(S_tot + 1) - 2s(s + 1)) / 2`, e.g. as the dimer probe
of [`task_spectrum`](@ref). For abelian symmetries use `MPSKitModels.S_exchange`.
"""
function spin_exchange_operator(elt::Type{<:Number}, ::Type{SU2Irrep} = SU2Irrep; spin = 1 // 2)
    P = SU2Space(spin => 1)
    SS = zeros(elt, P ⊗ P ← P ⊗ P)
    s = convert(Float64, spin)
    for (c, b) in blocks(SS)
        S = convert(Float64, c.j)
        b .= (S * (S + 1) - 2 * s * (s + 1)) / 2
    end
    return SS
end
spin_exchange_operator(::Type{SU2Irrep} = SU2Irrep; kwargs...) =
    spin_exchange_operator(ComplexF64, SU2Irrep; kwargs...)

"Number of sites an operator acts on."
_nsites(O::AbstractTensorMap) = numin(O)
_nsites(O::LocalMPO) = length(O)

"Excitation sector carried by the auxiliary (rightmost) leg of a local MPO."
function _excitation_sector(Ws::AbstractVector{<:MPSKit.MPOTensor})
    V = space(last(Ws), 4)'
    (length(sectors(V)) == 1 && dim(V, only(sectors(V))) == 1) ||
        throw(ArgumentError("the auxiliary leg of a local operator must carry a single sector with multiplicity 1, got $V"))
    return only(sectors(V))
end
_excitation_sector(O::Union{AbstractTensorMap, LocalMPO}) = _excitation_sector(local_mpo(O))

# left environment carrying the trivial left virtual leg of a local MPO
_left_unit_env(gs, site::Int, W::MPSKit.MPOTensor) =
    insertrightunit(MPSKit.l_LL(gs, site), 1; dual = isdual(space(W, 1)'))

"""
    _local_op_projection(Ws, gs, s::Int, left_gs = gs) -> (Φs, loc_ev)

Apply the local MPO `Ws` on sites `s, …, s + N - 1` of `gs` (in mixed gauge around `s`) and return
the effective one-site tensors `Φs[i][l p; u r]` at site `s + i - 1`, obtained by contracting all other
sites with `left_gs.AL` (left) and `left_gs.AR` (right). The leg `u` is the auxiliary leg of `W_N`,
which becomes the utility leg of the quasiparticle. Also returns `⟨gs|O|gs⟩` (zero for charged operators).
"""
function _local_op_projection(Ws::AbstractVector, gs, s::Int, left_gs = gs)
    N = length(Ws)
    sites = s:(s + N - 1)
    As = [i == 1 ? gs.AC[site] : gs.AR[site] for (i, site) in enumerate(sites)]

    # right environments R[x m; u y] of the sites right of i, open in the auxiliary leg u
    Vu = space(Ws[N], 4)'
    Rs = Vector{Any}(undef, N + 1)
    Rs[N + 1] = permute(MPSKit.r_RR(gs, last(sites)) ⊗ id(storagetype(Ws[N]), Vu), ((1, 2), (4, 3)))
    for i in N:-1:2
        A, W, Abar, R = As[i], Ws[i], left_gs.AR[sites[i]], Rs[i + 1]
        @tensor Rnew[-1 -2; -3 -4] := A[-1 1; 2] * W[-2 3; 1 4] * conj(Abar[-4 3; 5]) * R[2 4; -3 5]
        Rs[i] = Rnew
    end

    Φs = Vector{Any}(undef, N)
    GL = _left_unit_env(gs, s, Ws[1])
    for i in 1:N
        A, W, R = As[i], Ws[i], Rs[i + 1]
        @tensor Φ[-1 -2; -3 -4] := GL[-1 1; 2] * A[2 3; 4] * W[1 -2; 3 5] * R[4 5; -3 -4]
        Φs[i] = Φ
        i < N && (GL = GL * MPSKit.TransferMatrix(A, W, left_gs.AL[sites[i]]))
    end
    loc_ev = isunit(_excitation_sector(Ws)) ? inner(_with_util(gs.AC[s], Φs[1]), Φs[1]) : zero(scalartype(Φs[1]))
    return Φs, loc_ev
end

# `A[l p; r]` with the trivial utility leg `u` of `Φ[l p; u r]` inserted
function _with_util(A::MPSKit.MPSTensor, Φ::AbstractTensorMap)
    Au = insertleftunit(A, 3; dual = isdual(space(Φ, 3)'))
    space(Au) == space(Φ) || (Au = insertleftunit(A, 3; dual = !isdual(space(Φ, 3)')))
    space(Au) == space(Φ) || throw(SpaceMismatch("cannot match $(space(A)) to $(space(Φ))"))
    return Au
end
