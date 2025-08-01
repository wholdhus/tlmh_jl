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

function entropy(state, bond; spectrum=false)
	s_orth = orthogonalize(state, bond)
	U,S,V = svd(s_orth[bond], 
				(linkinds(s_orth, bond-1)..., 
				 siteinds(s_orth, bond)...))
	SvN = 0.0
	ps = zeros(dim(S, 1))
	for n=1:dim(S, 1)
		p = S[n,n]^2
		SvN -= p * log(p)
		ps[n] = p
	end
	if spectrum
		return SvN, ps
	else
		return SvN
	end
end


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
        println("Working on $fname")
        d = h5open(fname)
        newname = fname[1:end-3]*"_entropies.h5"
        
        dims = read(d, "dims")
        Lx = dims[1]
        nstates = read(d, "nstates")
        if "psi_$(nstates-1)" in keys(d)
            states = [read(d, "psi_$i", MPS) for i=0:nstates-1]
            energies = [read(d, "E_$i") for i=0:nstates-1]
            println("Energies:", energies)
            sinds = sortperm(energies)
            states =states[sinds]
            energies = energies[sinds]
            println("After sorting: ", energies)
            
            h5open(newname, "w") do fid
                fid["source"] = fname
                fid["dims"] = dims
                fid["energies"] = energies 
                fid["inds"] = ind_array(Lx, 2)
            end
            println("Created file $newname")

            println("Measuring entanglement entropy and spectrum")
            for (i, s) in enumerate(states)
                println("$(i-1)th state")
                #i = 1
                #s = states[1]
                entropies = zeros(Lx)
                spectra = Dict()
                for b = 1:Lx
                    entropies[b], spectra[b] = entropy(s, b; spectrum=true)
                end
                update_data(newname, "entropies$(i-1)", entropies)
                update_data(newname, "edge_spectrum$(i-1)", spectra[1])
                update_data(newname, "near_spectrum$(i-1)", spectra[Lx-1])
                update_data(newname, "center_spectrum$(i-1)", spectra[Lx])
                update_data(newname, "center-2_spectrum$(i-1)", spectra[Lx-2])
                update_data(newname, "center-3_spectrum$(i-1)", spectra[Lx-3])
                update_data(newname, "center-4_spectrum$(i-1)", spectra[Lx-4])
                println("Entropy:")
                println(entropies)
                println()
            end
        else
            println("Missing states! Skipping this one!!")
        end
        println("Done!")
    end
end

main()