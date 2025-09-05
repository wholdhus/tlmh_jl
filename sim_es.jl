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
        old_fnames = params["old_fnames"]
        fname = params["fname"]
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

        if "seed_fname" in keys(params)
            e, state = es_dmrg(old_fnames, fname, params["seed_fname"]; maxdim=maxdim, dmrg_params...)
        else
            e, state = es_dmrg(old_fnames, fname; maxdim=maxdim, dmrg_params...)
        end
        println("DMRG complete!")
        println("started from file $(ARG)")
        println("Energy: ")
        println(e)
    end
end

main()
