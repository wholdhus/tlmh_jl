include("dmrg.jl")

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
        if "g" in keys(params)
            g = params["g"]
        elseif "gs" in keys(params)
            g = params["gs"]
        else
            println("Not gonna work!")
        end

        parity = params["parity"]
        bc = params["bc"]

        dims = [Lx, Ly, parity, bc]
        println("Dimensions: ", dims)
        println("t = ", t)
        println("g = ")
        display(round.(g, digits=6))
        
        maxdim = params["maxdim"]
        if isa(maxdim, Number)
            maxdim=[min(div(maxdim, 6), 20), 
                    max(div(maxdim, 6), 20),
                    div(maxdim, 3), 
                    2*div(maxdim, 3), 
                    maxdim]
        end
        println("Maximum bond dimensions")
        display(maxdim)
        
        println("DMRG params")
        dmrg_params = params["dmrg_params"]
        dmrg_params = Dict(Symbol(k) => v for (k, v) in dmrg_params) # need to convert keys to symbols
        display(dmrg_params)
        
        if haskey(params, "datafile")
            energy, v = dmrg_e(t, g, dims, params["datafile"]; maxdim=maxdim, dmrg_params...)
        else
            energy, v = dmrg_e(t, g, dims; maxdim=maxdim, dmrg_params...)
        end
        println("DMRG complete!")
        println("g/gs: ")
        display(g)
        println("Energy/energies: ")
        display(energy)
    end
end

main()
