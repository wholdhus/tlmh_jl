println("Testing with a single g")
push!(empty!(ARGS), "tests/test_g.yml")
include("sim.jl")
println("")

println("Testing with a single g, low_weight")
push!(empty!(ARGS), "tests/test_g_weight.yml")
include("sim.jl")
println("")

println("Testing with anisotropy")
push!(empty!(ARGS), "tests/test_gv.yml")
include("sim.jl")
println("")

println("Testing with a single g and saving")
push!(empty!(ARGS), "tests/test_g_save.yml")
include("sim.jl")
println("")

println("Getting extra states from saved data")
push!(empty!(ARGS), "tests/test_g_more.yml")
include("resim.jl")
println()
