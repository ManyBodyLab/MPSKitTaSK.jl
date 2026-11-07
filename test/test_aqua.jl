using MPSKitTaSK
using MPSKit: MPSKit
using Aqua: Aqua
using Test
using TestExtras

@testset "Code quality (Aqua.jl)" begin
    # `MPSKit.timestep(::QP, …, ::TDVP)` is pirated as the reference for `timestep(…, ::TaSK)`;
    # it belongs upstream in MPSKit.
    Aqua.test_all(MPSKitTaSK; piracies = (; treat_as_own = [MPSKit.timestep]))
end
