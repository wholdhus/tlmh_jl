using ITensors, ITensorMPS
using HDF5
using Strided
using LinearAlgebra
using ArgParse
include("op_lists.jl")
include("utils.jl")

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
	a = MPS()
	b = MPO()
    ARG = ARGS[1]
    d = h5open(ARG)
    println("Opened $ARG")
    dims = read(d, "dims")
    Lx, Ly, parity, bc = dims
    states = Vector{MPS}()
    for i =0:10
        if "psi_$i" in keys(d)
            push!(states, read(d, "psi_$i", MPS))
        end
    end
    nstates = length(states)
    sites = siteinds(states[1])
    println("Found and read $nstates states!")
    
    t = read(d, "t")
    g = read(d, "g")
    println("Couplings: t=$t, g=$g")

    ladder_inds = ind_array(Lx, 2)

    
    opsums = [H0_OpSum(t, dims, ladder_inds),
              P1_OpSum(t, g, dims, ladder_inds),
              P2_OpSum(t, g, dims, ladder_inds),
              P3_OpSum(t, g, dims, ladder_inds)]
    mpos = [MPO(os, sites) for os in opsums]
    println("MPOs constructed!")
    expts = zeros(Float64, nstates, 4)
    names = ["H0", "P1", "P2", "P3"]
    for (i, s) in enumerate(states)
        println("$i th state")
        for (j, mpo) in enumerate(mpos)
            expts[i,j] = real(inner(s', mpo, s))
            name = names[j]
            e = expts[i,j]
            println("$name = $e")
        end
    end
end

main()