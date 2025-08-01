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
        
        old_fname = params["old_file"]
        nstates = params["nstates"]

        # optional args
        if "new_file" in keys(params)
            new_fname = params["new_file"]
        else
            new_fname = nothing
        end
        
        if "reconverge" in keys(params)
            reconverge = params["reconverge"]
        else
            reconverge = false
        end

        # optional new parameters for DMRG
        if "dmrg_params" in keys(params)
            dmrg_params = Dict(Symbol(k) => v for (k, v) in params["dmrg_params"])
        else
            dmrg_params = Dict()
        end
        
        energy, v = redmrg(old_fname, nstates;
                           reconverge=reconverge, 
                           newname=new_fname, dmrg_params...)
        println("DMRG complete!")
        println("Energy/energies: ")
        display(energy)
    end
end

main() 