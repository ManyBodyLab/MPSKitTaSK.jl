"""
    tangent_variance_2site(H, phi::MPSKit.QP, lenvs = ..., renvs = ...; envs = nothing, alg_environments = MPSKit.Defaults.alg_environments())

Compute the 2-site tangent-space leakage variance σ^{2⊥} = || P^{2⊥}_{ψ_g} H |Φ⟩ ||^2 (Eq. S18 of arXiv:2510.07279v1).
"""
function tangent_variance_2site(
        H, phi::MPSKit.QP,
        lenvs = MPSKit.environments(phi.left_gs, H, phi.left_gs),
        renvs = MPSKit.istopological(phi) ? MPSKit.environments(phi.right_gs, H, phi.right_gs) : lenvs;
        envs = nothing,
        alg_environments = MPSKit.Defaults.alg_environments()
    )
    if !isnothing(envs)
        lenvs = envs
        if !MPSKit.istopological(phi)
            renvs = envs
        end
    end
    qpenvs = MPSKit.environments(phi, H, phi, alg_environments; lenvs = lenvs, renvs = renvs)
    L = length(phi)
    TR = real(scalartype(phi))
    var_sum = zero(TR)

    for l in 1:(L - 1)
        VL = phi.VLs[l]
        AR_next = phi.right_gs.AR[l + 1]
        VR = MAK.right_null!(MPSKit._transpose_tail(AR_next; copy = true))

        if dim(domain(VL)) == 0 || dim(codomain(VR)) == 0
            continue
        end

        GL = MPSKit.leftenv(lenvs, l, phi.left_gs)
        GR = MPSKit.rightenv(renvs, l + 1, phi.right_gs)

        L_ex = GL * MPSKit.TransferMatrix(phi[l], H[l], VL)
        if l > 1
            L_ex += qpenvs.leftBenvs[l] * MPSKit.TransferMatrix(phi.right_gs.AR[l], H[l], VL)
        end
        L_gs = GL * MPSKit.TransferMatrix(phi.left_gs.AL[l], H[l], VL)

        B_next = phi[l + 1]
        AL_next = phi.left_gs.AL[l + 1]
        W_next = H[l + 1]

        @plansor Φ2[-1 -2; -3 -4] := L_gs[-1 5; 4] * B_next[4 2; -3 1] * W_next[5 -2; 2 3] * GR[1 3; -4]
        @plansor Φ2[-1 -2; -3 -4] += L_ex[-1 4; -3 5] * AR_next[5 2; 1] * W_next[4 -2; 2 3] * GR[1 3; -4]
        if l + 1 < L
            GBR = qpenvs.rightBenvs[l + 1]
            @plansor Φ2[-1 -2; -3 -4] += L_gs[-1 2; 1] * AL_next[1 3; 4] * W_next[2 -2; 3 5] * GBR[4 5; -3 -4]
        end

        @tensor Y[-1; -2 -3] := Φ2[-1 1; -2 2] * conj(VR[-3; 2 1])
        var_sum += real(dot(Y, Y))
    end
    return var_sum
end
