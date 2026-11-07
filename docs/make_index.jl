using Literate: Literate
using MPSKitTaSK

Literate.markdown(
    joinpath(pkgdir(MPSKitTaSK), "docs", "files", "README.jl"),
    joinpath(pkgdir(MPSKitTaSK), "docs", "src");
    flavor = Literate.DocumenterFlavor(),
    name = "index",
)
