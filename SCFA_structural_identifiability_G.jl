# SCFA-T2D: structural identifiability of ALL 22 kinetic parameters
# Observed output: y(t) = G(t) only
# Initial conditions are NOT included in funcs_to_check.
# The model assumes generic UNKNOWN initial conditions by default.
#
# First-time install:
#   julia -e 'using Pkg; Pkg.add("StructuralIdentifiability")'
#
# Local analysis (recommended first):
#   julia SCFA_structural_identifiability_Gin_G_only.jl
# Optional global analysis (may be computationally expensive):
#   julia SCFA_structural_identifiability_Gin_G_only.jl --global

using StructuralIdentifiability
using Printf

# States: G = glucose, I = insulin, B = beta-cell mass, S = SCFA
# G_in = G_meal + G_renal is one combined glucose input parameter (Gin).
# The quartic bump A/(1+((G-Gc)/w)^4) is represented by
# A*w^4/(w^4+(G-Gc)^4) to keep the RHS rational.
model = @ODEmodel(
    G'(t) = Gin
            + Ghep/((1 + alphaS*S(t))*(1 + kI*I(t)))
            - Sg*G(t)
            - SI0*(1 + etaS*S(t)/(KS + S(t)))*I(t)*G(t),

    I'(t) = sigmaG*(G(t)^2/(alphaG + G(t)^2))
            *(1 + gammaS*S(t)/(KGLP + S(t)))*B(t)
            - di*I(t),

    B'(t) = (A*w^4/(w^4 + (G(t) - Gc)^4)
             - d0 + rhoS*S(t)/(Kbeta + S(t)))*B(t),

    S'(t) = PS/(1 + cG*G(t)) - kS*S(t),

    y(t) = G(t)
)

# Check parameters only: do not include states or initial conditions.
params = sort!(collect(model.parameters), by=string)

println("States:            G, I, B, S")
println("Observation:       y(t) = G(t)")
println("Initial values:    not assessed; generic unknown ICs")
println("Parameter count:   ", length(params))
println("Parameters:        ", join(string.(params), ", "))
@assert length(params) == 22 "Expected 22 parameters after replacing Gmeal + Grenal by Gin"

println("\n===== LOCAL STRUCTURAL IDENTIFIABILITY =====")
local_result = assess_local_identifiability(
    model;
    funcs_to_check=params,
    prob_threshold=0.99
)

local_path = joinpath(@__DIR__, "SCFA_Gin_identifiability_local.csv")
open(local_path, "w") do io
    println(io, "parameter,locally_identifiable")
    for p in params
        yes = local_result[p]
        @printf("%-18s %s\n", string(p), yes ? "YES" : "NO")
        println(io, string(p), ",", yes)
    end
end
println("Saved: ", local_path)

# Global structural identifiability is optional.
if "--global" in ARGS
    println("\n===== GLOBAL STRUCTURAL IDENTIFIABILITY =====")
    global_result = assess_identifiability(
        model;
        funcs_to_check=params,
        prob_threshold=0.99
    )

    global_path = joinpath(@__DIR__, "SCFA_Gin_identifiability_global.csv")
    open(global_path, "w") do io
        println(io, "parameter,identifiability")
        for p in params
            status = global_result[p]
            @printf("%-18s %s\n", string(p), string(status))
            println(io, string(p), ",", status)
        end
    end
    println("Saved: ", global_path)
else
    println("\nLocal analysis completed.")
    println("Use --global for the optional global analysis.")
end
