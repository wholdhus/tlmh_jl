using ITensors
using ITensorMPS
using Random
using ProgressBars
using HDF5
using UUIDs

include("utils.jl")
include("op_lists.jl")

ITensors.op(::OpName"P",::SiteType"Fermion") =
	 [-1 0
	  0  1];

mutable struct DMRGSizeObserver <: AbstractObserver
    energy_tol::Float64
    force_gc::Bool
    last_energy::Float64
    DMRGSizeObserver(energy_tol=0.0,force_gc=false) = new(energy_tol,force_gc,1000.0)
end

function ITensorMPS.checkdone!(o::DMRGSizeObserver;kwargs...)
    sw = kwargs[:sweep]
    energy = kwargs[:energy]
    de = abs(energy-o.last_energy)/abs(energy)
    if de < o.energy_tol
      println("Stopping DMRG after sweep $sw due to relative energy difference $de")
      return true
    end
    # Otherwise, update last_energy and keep going
    o.last_energy = energy
    return false
  end

function ITensorMPS.measure!(o::DMRGSizeObserver; bond, half_sweep, psi, projected_operator, kwargs...)
    if bond==1 && half_sweep==2
        # psi_size =  Base.format_bytes(Base.summarysize(psi))
        # PH_size =  Base.format_bytes(Base.summarysize(projected_operator))
        # println("|psi| = $psi_size, |PH| = $PH_size")
        if o.force_gc
            maxRss = Base.format_bytes(Sys.maxrss())
            println("Maxrss size: $maxRss")
            GC.gc()
            maxRss = Base.format_bytes(Sys.maxrss())
            println("After GC.gc(): $maxRss")
        end
    end
end


"""
    update_data(fname, key, obj; outputlevel=1, group=nothing)

TBW
"""
function update_data(fname, key, obj; outputlevel=1, group=nothing)
    h5open(fname, "cw") do fid
        dset = fid
        if ! isnothing(group)
            dset = fid[group]
        end
        if haskey(dset, key)
            printif("Data already exists?", outputlevel)
        else
            dset[key] = obj
        end
    end
    printif("Data with key $(key) saved to $(fname) in group $(group)!", outputlevel)	
end


"""
    replace_data(fname, key, obj)

TBW
"""
function replace_data(fname, key, obj; outputlevel=1)
    h5open(fname, "cw") do fid
        delete_object(fid, key)
        fid[key] = obj
    end
    printif("Data with key $(key) replace in $(fname)!", outputlevel)	
end


function get_psi0(dims, sites)
    Lx, Ly, parity, bc = dims
    Lx = Int(Lx)
    Ly = Int(Ly)
    parity = Int(parity)
    N = Lx*Ly
    state = ["0" for n=1:N]
    if parity == -1
        println("Odd parity")
        state[1] = "1"
    else
        println("Even parity")
    end
    psi0 = random_mps(sites, state, linkdims=2)
    return psi0
end

function gs_dmrg(t::Number, g::Number, dims::Vector,
                 fname::String; 
                 nsweeps::Int=10,
                 maxdim::Vector{Int}=[10, 20, 100, 200],
                 energy_tol::Float64=1E-6,
                 cutoff::Float64=1E-7, 
                 outputlevel::Int=1,
                 kdim::Int=8, 
                 noise::Vector{Float64}=[0.0], 
                 force_maxdim::Bool=false,
                 force_gc::Bool=false, 
                 simple_inds::Bool=true,
                 save_H::Bool=false)
    Lx, Ly, parity, bc = dims
    Lx = Int(Lx)
    Ly = Int(Ly)
    parity = Int(parity)
    N = Lx*Ly
    mindim = 2
    if force_maxdim
        println("Forcing bond dimensions to be")
        mindim = maxdim
        println(mindim)
    end
    sites = siteinds("Fermion",N;conserve_nfparity=true)
    psi0 = get_psi0(dims, sites)
    inds = ind_array(Lx, Ly; simple=simple_inds)    
    observer = DMRGSizeObserver(energy_tol, force_gc)
    printif("Initializing hdf5 file", outputlevel)
    h5open(fname, "w") do fid
        fid["dims"] = dims
        fid["inds"] = dims
        fid["maxdim"] = maxdim
        fid["energy_tol"] = energy_tol
        fid["noise"] = noise
        fid["kdim"] = kdim
        fid["nsweeps"] = nsweeps
        fid["cutoff"] = cutoff
        fid["g"] = g
        fid["t"] = t
        fid["simple_inds"] = simple_inds
    end
    printif("Using g=$(g)", outputlevel)
    os = H_OpSum(t, g, dims, inds)
    H = MPO(os, sites)
    printif("Interacting MPO constructed!", outputlevel)
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff,
                                  mindim=mindim,
				                  observer=observer, 
				                  outputlevel=outputlevel,
				                  noise=noise,
				                  eigsolve_krylovdim=kdim)
    if save_H
        update_data(fname, "H", H)
    end
    update_data(fname, "energy", energy)
    update_data(fname, "psi", psi)
    return energy, psi
end

function es_dmrg(old_fnames::Vector{String},
                 fname::String; 
                 nsweeps::Int=10,
                 maxdim::Vector{Int}=[10, 20, 100, 200],
                 energy_tol::Float64=1E-6,
                 cutoff::Float64=1E-7, 
                 outputlevel::Int=1,
                 kdim::Int=8, 
                 noise::Vector{Float64}=[0.0], 
                 force_maxdim::Bool=false,
                 force_gc::Bool=false,
                 weight::Number=1000)
    mindim = 2
    if force_maxdim
        println("Forcing bond dimensions to be")
        mindim = maxdim
        println(mindim)
    end
    observer = DMRGSizeObserver(energy_tol, force_gc)
    fid = h5open(old_fnames[1])
    t = read(fid, "t")
    g = read(fid, "g")
    dims = read(fid, "dims")
    Lx = Int(dims[1])
    Ly = Int(dims[2])
    simple_inds = read(fid, "simple_inds")
    inds = ind_array(Lx, Ly; simple=simple_inds)    
    printif("From first file: t=$t, g=$g", outputlevel)
    printif("dims = $(dims)", outputlevel)
    printif("Opening old state files", outputlevel)
    psis = [MPS() for ofn in old_fnames]
    for (i, ofn) in enumerate(old_fnames)
        fid = h5open(ofn, "r")
        Ef = read(fid, "energy")
        printif("file: $(ofn)", outputlevel)
        printif("energy: $(Ef)", outputlevel)
        psis[i] = read(fid, "psi", MPS)
    end
    sites = siteinds(psis[1])
    psi0 = get_psi0(dims, sites)
    printif("Initializing hdf5 file", outputlevel)
    h5open(fname, "w") do fid
        fid["dims"] = dims
        fid["inds"] = inds
        fid["maxdim"] = maxdim
        fid["energy_tol"] = energy_tol
        fid["noise"] = noise
        fid["kdim"] = kdim
        fid["nsweeps"] = nsweeps
        fid["cutoff"] = cutoff
        fid["g"] = g
        fid["t"] = t
        fid["other_statefiles"] = old_fnames
        fid["simple_inds"] = simple_inds
    end
    printif("Using g=$(g)", outputlevel)
    os = H_OpSum(t, g, dims, inds)
    H = MPO(os, sites)
    printif("Interacting MPO constructed!", outputlevel)
    _, psi = dmrg(H, psis, psi0; nsweeps, maxdim, cutoff,
                       mindim=mindim,
			           observer=observer, 
			           outputlevel=outputlevel,
			           noise=noise,
				       eigsolve_krylovdim=kdim,
                       weight=weight)
    energy = real(inner(psi', H, psi))   
    update_data(fname, "energy", energy)
    update_data(fname, "psi", psi)
    return energy, psi
end