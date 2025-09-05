include("dmrg2.jl")

using LinearAlgebra
using Strided
using YAML
using ArgParse


function main()
    
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
        Strided.set_num_threads(1)
        println("Using threaded blocksparse with ", Threads.nthreads(), " threads!")
        ITensors.enable_threaded_blocksparse(true)
    end
    println("All args:")
    display(ARGS)
    for ARG in ARGS
        params = YAML.load_file(ARG)
        println("Params: ")
        display(params)
        
        # dimensions, couplinks
        Lx = params["Lx"]
        Ly = params["Ly"]
        t = params["t"]
        g = params["g"]
        parity = params["parity"]
        bc = params["bc"]
        dims = [Lx, Ly, parity, bc]
        println("Dimensions: ", dims)
        println("t = ", t)
        println("g = ")
        display(g)
        
        maxdim = params["maxdim"]
        if isa(maxdim, Number)
            maxdim=[min(div(maxdim, 6), 20),
                    max(div(maxdim, 6), 20),
                    div(maxdim, 3),
                    2*div(maxdim, 3),   
                    2*div(maxdim, 3), 
                    maxdim]
        end
        println("Maximum bond dimensions")
        display(maxdim)
        
        println("DMRG params")
        dmrg_params = params["dmrg_params"]
        dmrg_params = Dict(Symbol(k) => v for (k, v) in dmrg_params) # need to convert keys to symbols
        display(dmrg_params)
        fname = params["fname"]
        e, state = gs_dmrg(t, g, dims, fname; maxdim=maxdim, dmrg_params...)

        println("DMRG complete!")
        println("g: ")
        println(g)
        println("Energy: ")
        println(e)
    end
end

main()
