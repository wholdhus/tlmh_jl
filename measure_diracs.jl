using ITensors
using ITensorMPS
using HDF5
using Strided
using LinearAlgebra
include("utils.jl")
include("dmrg.jl")
ITensors.op(::OpName"P",::SiteType"Fermion") =
[-1 0
 0  1];


function main()
    
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
        Strided.set_num_threads(1)
        println("Using threaded blocksparse with ", Threads.nthreads(), " threads!")
        ITensors.enable_threaded_blocksparse(true)
    end

    println("All args:")
    display(ARGS)
    for fname in ARGS
        d = h5open(fname)
        newname = fname[1:end-3]*"_diracs.h5"
        
        dims = read(d, "dims")
        Lx = dims[1]
        nstates = read(d, "nstates")
        states = [read(d, "psi_$i", MPS) for i=0:nstates-1]
        energies = [read(d, "E_$i") for i=0:nstates-1]
        x0 = div(Lx, 2)

        h5open(newname, "w") do fid
            fid["source"] = fname
            fid["dims"] = dims
            fid["energies"] = energies 
            fid["x0"] = x0
            fid["inds"] = ind_array(Lx, 2)
        end
        println("Created file $newname")

        println("Measuring fermion correlations")
        for (i,s) in enumerate(states)
            println("$(i-1)th state")
            update_data(newname, "nn$(i-1)", correlation_matrix(s, "N", "N"))
            update_data(newname, "n$(i-1)", expect(s, "N"))
            update_data(newname, "pp$(i-1)", correlation_matrix(s, "P", "P"))
            update_data(newname, "p$(i-1)", expect(s, "P"))
            update_data(newname, "cdcd$(i-1)", correlation_matrix(s, "Cdag", "Cdag"))
            update_data(newname, "cdc$(i-1)", correlation_matrix(s, "Cdag", "C"))
        end
        println("Done!")
    end
end

main()