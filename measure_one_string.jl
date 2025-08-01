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


function string_corr_ops(N, sites, i0)
	names = ["PP", "PM"]
	tails = ["Cdag", "C"]
	ops = Dict()
	for name in names
		ops[name] = [MPO() for i = (i0+1):N]
	end
	# sites = siteinds(states[1])
	for i = i0+1:N
		#println(i)
		# scrap = [r1[haf]]
		lsts = [[1.0,"Cdag",i0],
				[1.0,"Cdag",i0]]
		for j = (i0):(i-1)
			#println(j)
			for lst in lsts
				push!(lst, "P", j)
			end
		end
		for j = 1:2
			lst = lsts[j]
			tail = tails[j]
			name = names[j]
			push!(lst, tail, i)
			#println(name)
			#println(lst)
			tpl = tuple(lst...)
			os = OpSum()
			add!(os, tpl)
			# println("OpSum:", os)
			ops[name][i-i0] = MPO(os, sites)
		end
	end
    return ops
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
        d = h5open(fname)
        newname = fname[1:end-3]*"_string.h5"
        
        dims = read(d, "dims")
        Lx = dims[1]
        N = 2*Lx
        nstates = read(d, "nstates")
        states = [read(d, "psi_$i", MPS) for i=0:nstates-1]
        energies = [read(d, "E_$i") for i=0:nstates-1]
        println("Energies:", energies)
        sinds = sortperm(energies)
        states =states[sinds]
        energies = energies[sinds]
        println("After sorting: ", energies)

        bc = dims[4]
        if bc == 0
            i0 = Lx
        else
            i0 = 1
        end
        println("i0 = $(i0)")

        h5open(newname, "w") do fid
            fid["source"] = fname
            fid["dims"] = dims
            fid["energies"] = energies 
            fid["inds"] = ind_array(Lx, 2)
            fid["i0"] = i0
        end
        println("Created file $newname")
        nstates = length(states)
        pp_corrs = zeros(ComplexF64, nstates, N)
        pm_corrs = zeros(ComplexF64, nstates, N)




        ops = string_corr_ops(N, siteinds(states[1]), i0)
        
        println("ops formed")
        for (i,s) in enumerate(states)
            #println("Measuring $(i-1)th state")            
            pp_corrs[i, i0+1:end] = [inner(s', op, s) for op in ops["PP"]]
            pm_corrs[i, i0+1:end] = [inner(s', op, s) for op in ops["PM"]]
        end
        println("ground state s+s+")
        println(round.(real(pp_corrs[1, :]), digits=3))
        println("ground state s+s-")
        println(round.(real(pm_corrs[1, :]), digits=3))
        println("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        println()
        update_data(newname, "s+s+", pp_corrs)
        update_data(newname, "s+s-", pm_corrs)
        print("Finished!")
    end
end

main()