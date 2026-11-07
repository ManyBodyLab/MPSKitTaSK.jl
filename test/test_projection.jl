using Test
using TensorKit: @tensor
using VectorInterface: inner, add, scale
include("setup_models.jl")

# hand-coded one- and two-site projections of the original TaSK implementation, used as reference
function reference_projection(op, gs, pos)
    if numin(op) == 1
        @tensor ϕ[-1 -2; -3] := gs.AC[pos][-1 1; -3] * op[-2; 1]
        return ϕ, nothing, inner(gs.AC[pos], ϕ)
    end
    @tensor T[-1 -2 -3; -4] := gs.AC[pos][-1 -2; 1] * gs.AR[pos + 1][1 -3; -4]
    @tensor T_op[-1 -2 -3; -4] := T[-1 1 2; -4] * op[-2 -3; 1 2]
    @tensor ϕ_dir[-1 -2; -3] := T_op[-1 -2 1; 2] * conj(gs.AR[pos + 1][-3 1; 2])
    @tensor ϕ_car[-1 -2; -3] := conj(gs.AL[pos][1 2; -1]) * T_op[1 2 -2; -3]
    return ϕ_dir, ϕ_car, inner(gs.AC[pos], ϕ_dir)
end

@testset "local MPO projection" begin
    H, gs, = finite_heisenberg(Trivial)
    Sz = S_z(ComplexF64, Trivial)
    SS = S_exchange(ComplexF64, Trivial)

    @testset "matches hand-coded reference ($(numin(op))-site, site $s)" for (op, s) in ((Sz, 3), (SS, 3), (SS, 5))
        Φs, ev = MPSKitTaSK._local_op_projection(local_mpo(op), gs, s)
        ϕ_dir, ϕ_car, ev_ref = reference_projection(op, gs, s)
        @test Φs[1] ≈ MPSKitTaSK._with_util(ϕ_dir, Φs[1]) atol = 1.0e-12
        isnothing(ϕ_car) || @test Φs[2] ≈ MPSKitTaSK._with_util(ϕ_car, Φs[2]) atol = 1.0e-12
        @test ev ≈ ev_ref atol = 1.0e-12
        @test ev ≈ expectation_value(gs, Tuple(s:(s + numin(op) - 1)) => op) atol = 1.0e-12
    end

    @testset "three-site operator" begin
        O3 = SS ⊗ id(ℂ^2)
        Φs, ev = MPSKitTaSK._local_op_projection(local_mpo(O3), gs, 2)
        @test length(Φs) == 3
        @test ev ≈ expectation_value(gs, (2, 3, 4) => O3) atol = 1.0e-12
        _, b0, expval = project_tangent(O3, 2, gs)
        @test b0 > 0
        @test expval ≈ ev atol = 1.0e-12
    end

    @testset "single-site norm" begin
        # a one-site operator acting on the orthogonality centre lies entirely in the tangent space
        S, mult = spin_probe(SU2Irrep; spin = 1 // 2)
        _, b0, = project_tangent(S, 3, finite_heisenberg(SU2Irrep)[2])
        @test b0^2 ≈ 3 / 4
        _, b0z, = project_tangent(Sz, 3, gs)
        @test b0z^2 ≈ 1 / 4
    end
end

@testset "momentum interface" begin
    H, gs, = finite_heisenberg(SU2Irrep)
    S, = spin_probe(SU2Irrep; spin = 1 // 2)
    L = length(gs)
    k = 2π * 2 / L
    b0s = [
        project_tangent(S, gs, k)[2], project_tangent(S, gs; k)[2], project_tangent(S, gs; momentum = k)[2],
        project_tangent(S, gs; amplitudes = [cis(k * j) / sqrt(L) for j in 1:L])[2],
    ]
    @test all(≈(b0s[1]), b0s)
    @test project_tangent(S, gs, k; site = 2)[2] ≈ project_tangent(S, 2, gs)[2]
    @test project_tangent(S, gs)[2] ≈ project_tangent(S, cld(L, 2), gs)[2]
    @test project_tangent(S, gs, 0.0)[2] < 1.0e-8   # the total spin annihilates the singlet
    @test_throws ArgumentError project_tangent(S, gs, 0.3)
    @test_throws ArgumentError project_tangent(H, gs, π / 2)

    # the static structure factor is symmetric under k → 2π - k
    S_k = [project_tangent(S, gs, 2π * n / L)[2] for n in 1:(L - 1)]
    @test S_k ≈ reverse(S_k)
end

@testset "projection of MPOs and argument checks" begin
    H, gs, = finite_heisenberg(Trivial)
    L = length(gs)
    Sz = S_z(ComplexF64, Trivial)

    # H|gs⟩ = E₀|gs⟩ has no component orthogonal to |gs⟩
    _, b0, ev = project_tangent(H, gs)
    @test ev ≈ expectation_value(gs, H) atol = 1.0e-8
    @test b0 < 1.0e-4
    @test project_tangent(H, gs, 2π)[3] ≈ ev

    @test_throws BoundsError project_tangent(Sz, L + 1, gs)
    @test_throws BoundsError project_tangent(Sz, gs, 2π / L; site = 0)
    @test_throws ArgumentError project_tangent(Sz, gs, 1.0)

    # the projection is linear in the amplitudes
    amps = ComplexF64[0.3, 0, -0.5im, 0, 0, 0.2]
    qp, b0, = project_tangent(Sz, gs; amplitudes = amps)
    ref = nothing
    for s in findall(!iszero, amps)
        qp_s, b0_s, = project_tangent(Sz, s, gs)
        term = scale(qp_s, amps[s] * b0_s)
        ref = isnothing(ref) ? term : add(ref, term)
    end
    @test norm(scale(qp, b0) - ref) < 1.0e-10
end
