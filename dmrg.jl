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
    ent_entropy(psi, b; return_spectrum=false)

TBW
"""
function ent_entropy(psi, b; return_spectrum=false)
    s = orthogonalize(psi, b)
	U,S,V = svd(s[b], (linkinds(s, b-1)..., siteinds(s, b)...))
	SvN = 0.0
	for n=1:dim(S, 1)
  		p = S[n,n]^2
  		SvN -= p * log(p)
	end
	if return_spectrum
	    return SvN, p
	else
	    return SvN
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


function get_psi0s(dims, sites, nstates, seed)

    Lx, Ly, parity, bc = dims
    Lx = Int(Lx)
    Ly = Int(Ly)
    parity = Int(parity)
    N = Lx*Ly
    if seed == "af" || seed == "fm"
        if seed == "af"
            state = [isodd(n) ? "0" : "1" for n=1:N]
        else
            state = ["0" for n=1:N]
        end
        if parity == 1
            state[1] = "1"
        end
        psi0s = [MPS() for n=1:nstates]
        if seed == "af" || seed == "fm"
            println("Using nonrandomized state for psi0")
            psi0s[1] = MPS(sites, state)
            if nstates > 1
                println("Shuffling for excited state seeds")
                for n=2:nstates
                    psi0s[n] = MPS(sites, shuffle(state))
                end
            end
        end
    else
        state = ["0" for n=1:N]
        if parity == 1
            state[1] = "1"
        end
        psi0s = [random_mps(sites, state, linkdims=4) for n=1:nstates]
    end
    return psi0s
end


# Better: use dims::Tuple{Int64, Int64, Int64, Number} (or Float)
function dmrg_e(t::Number, g::Number, dims::Vector;
                seed::String="rand",
                nstates::Int=1, nsweeps::Int=10,
                maxdim::Vector{Int}=[10, 20, 100, 200],
                energy_tol::Float64=1E-11,
                cutoff::Float64=1E-12, outputlevel::Int=1,
                kdim::Int=8, use_noise::Bool=false, force_maxdim::Bool=false,
                force_gc::Bool=false, weight::Number=1000.0,
                simple_inds::Bool=false)
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
    inds = ind_array(Lx, Ly; simple=simple_inds)    


    energies = zeros(Float64, nstates)
    os = H_OpSum(t, g, dims, inds)
    H = MPO(os, sites)
    observer = DMRGSizeObserver(energy_tol, force_gc)
    if use_noise
        noise = [1E-5, 1E-6, 1E-7, 1E-8, 10*energy_tol, energy_tol]
    else
        noise = [0.0]
    end
    psi0s = get_psi0s(dims, sites, nstates, seed)
    psis = [MPS() for n=1:nstates]
    energies[1], psis[1] = dmrg(H, psi0s[1]; nsweeps, maxdim, cutoff,
                            mindim=mindim,
				            observer=observer, 
				            outputlevel=outputlevel,
				            noise=noise,
				            eigsolve_krylovdim=kdim
				            )
	if nstates > 1
	    printif("", outputlevel)
	    printif("Finding excited states!", outputlevel)
	    for n=2:nstates
		    _, psis[n] = dmrg(H, psis[1:n-1], psi0s[n];
						      nsweeps, maxdim, cutoff,
                              mindim=mindim,
				              observer=observer, 
				              outputlevel=outputlevel,
				              #weight=2*abs(energies[1]),
				              weight=weight,
                              noise=noise,
				              eigsolve_krylovdim=kdim)
		    energies[n] = real(inner(psis[n]', H, psis[n]))
		
	    end
	end
    printif("Overlaps: ", outputlevel)
    for n = 1:nstates
        for m = 1:n
            printif("<$(n-1)|$(m-1)>: $(abs(inner(psis[n]', psis[m])))", outputlevel)
        end
    end
    printif("", outputlevel)
    return energies, psis
end


function dmrg_e(t::Number, gs::Vector, dims::Vector,
                fname::String;
                seed::String="rand",
                nstates::Int=1, nsweeps::Int=10,
                maxdim::Vector{Int}=[10, 20, 100, 200],
                energy_tol::Float64=1E-6,
                cutoff::Float64=1E-7, outputlevel::Int=1,
                kdim::Int=8, use_noise::Bool=false, force_maxdim::Bool=false,
                force_gc::Bool=false, weight::Number=1000, simple_inds::Bool=false)
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
    psi0s = get_psi0s(dims, sites, nstates, seed)
    psis = [MPS() for n=1:nstates]

    inds = ind_array(Lx, Ly; simple=simple_inds)    
    energies = zeros(Float64, nstates)
    observer = DMRGSizeObserver(energy_tol, force_gc)
    if use_noise
        noise = [1E-5, 1E-6, 1E-8, 10*energy_tol, energy_tol]
    else
        noise = [0.0]
    end
    printif("Initializing hdf5 file", outputlevel)
    h5open(fname, "w") do fid
        fid["dims"] = dims
        fid["maxdim"] = maxdim
        fid["energy_tol"] = energy_tol
        fid["noise"] = noise
        fid["kdim"] = kdim
        fid["nstates"] = nstates
        fid["nsweeps"] = nsweeps
        fid["cutoff"] = cutoff
        fid["g"] = gs
        fid["t"] = t
    end
    printif("Using g=$(gs)", outputlevel)
    os = H_OpSum_asym(t, gs, dims, inds)
    H = MPO(os, sites)
    printif("Interacting MPO constructed!", outputlevel)
    energies[1], psis[1] = dmrg(H, psi0s[1]; nsweeps, maxdim, cutoff,
                                  mindim=mindim,
				                  observer=observer, 
				                  outputlevel=outputlevel,
				                  noise=noise,
				                  eigsolve_krylovdim=kdim
				                  )
    update_data(fname, "H", H)
    update_data(fname, "E_0", energies[1])
    update_data(fname, "psi_0", psis[1])
	if nstates > 1
	    printif("", outputlevel)
	    printif("Finding excited states!", outputlevel)
	    for n=2:nstates
		    _, psis[n] = dmrg(H, psis[1:n-1], psi0s[n];
						      nsweeps, maxdim, cutoff,
                              mindim=mindim,
				              observer=observer, 
				              outputlevel=outputlevel,
				              weight=weight,
				              noise=noise,
				              eigsolve_krylovdim=kdim
				              )
		    energies[n] = real(inner(psis[n]', H, psis[n]))
            update_data(fname, "E_$(n-1)", energies[n])
            update_data(fname, "psi_$(n-1)", psis[n])
		    printif("", outputlevel)
	    end
    end
    printif("Overlaps: ", outputlevel)
    for n = 1:nstates
        for m = 1:n
            printif("<$(n-1)|$(m-1)>: $(abs(inner(psis[n]', psis[m])))", outputlevel)
        end
    end
    printif("", outputlevel)
    return energies, psis
end


function dmrg_e(t::Number, g::Number, dims::Vector,
                fname::String;
                seed::String="rand",
                nstates::Int=1, nsweeps::Int=10,
                maxdim::Vector{Int}=[10, 20, 100, 200],
                energy_tol::Float64=1E-6,
                cutoff::Float64=1E-7, outputlevel::Int=1,
                kdim::Int=8, use_noise::Bool=false, force_maxdim::Bool=false,
                force_gc::Bool=false, weight::Number=1000.0, simple_inds::Bool=false)
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
    psi0s = get_psi0s(dims, sites, nstates, seed)
    psis = [MPS() for n=1:nstates]
    printif("", outputlevel)

    inds = ind_array(Lx, Ly; simple=simple_inds)    
    energies = zeros(Float64, nstates)
    observer = DMRGSizeObserver(energy_tol, force_gc)
    if use_noise
        noise = [1E-5, 1E-6, 1E-8, 10*energy_tol, energy_tol]
    else
        noise = [0.0]
    end
    printif("Initializing hdf5 file", outputlevel)
    h5open(fname, "w") do fid
        fid["dims"] = dims
        fid["maxdim"] = maxdim
        fid["energy_tol"] = energy_tol
        fid["noise"] = noise
        fid["kdim"] = kdim
        fid["nstates"] = nstates
        fid["nsweeps"] = nsweeps
        fid["cutoff"] = cutoff
        fid["g"] = g
        fid["t"] = t
    end

    os = H_OpSum(t, g, dims, inds)
    H = MPO(os, sites)
    printif("Interacting MPO constructed!", outputlevel)
    energies[1], psis[1] = dmrg(H, psi0s[1]; nsweeps, maxdim, cutoff,
                                  mindim=mindim,
				                  observer=observer, 
				                  outputlevel=outputlevel,
				                  noise=noise,
				                  eigsolve_krylovdim=kdim
				                  )
    update_data(fname, "H", H)
    update_data(fname, "E_0", energies[1])
    update_data(fname, "psi_0", psis[1])
	if nstates > 1
	    printif("", outputlevel)
	    printif("Finding excited states!", outputlevel)
	    for n=2:nstates
		    _, psis[n] = dmrg(H, psis[1:n-1], psi0s[n];
						      nsweeps, maxdim, cutoff,
                              mindim=mindim,
				              observer=observer, 
				              outputlevel=outputlevel,
				              weight=weight,
				              noise=noise,
				              eigsolve_krylovdim=kdim
				              )
		    energies[n] = real(inner(psis[n]', H, psis[n]))
            update_data(fname, "E_$(n-1)", energies[n])
            update_data(fname, "psi_$(n-1)", psis[n])
		    printif("", outputlevel)
	    end
    end
    printif("Overlaps: ", outputlevel)
    for n = 1:nstates
        for m = 1:n
            printif("<$(n-1)|$(m-1)>: $(abs(inner(psis[n]', psis[m])))", outputlevel)
        end
    end
    printif("", outputlevel)
    return energies, psis
end


function redmrg(fname::String, nstates::Int;
                reconverge::Bool=true,
                newname::Union{String, Nothing}=nothing,
                nsweeps::Union{Int, Nothing}=nothing,
                maxdim::Union{Vector{Int}, Nothing}=nothing,
                energy_tol::Union{Float64, Nothing}=nothing,
                cutoff::Union{Float64, Nothing}=nothing, 
                outputlevel::Int=1,
                kdim::Union{Int, Nothing}=nothing,
                force_gc::Bool=false, weight::Number=1000.0, simple_inds::Bool=false)
    printif("Reading from hdf5 file $(fname)", outputlevel)
    fid = h5open(fname, "r")
    dims = read(fid, "dims")
    noise = read(fid, "noise")
    nstates0 = read(fid, "nstates")
   
    psis0 = [read(fid, "psi_$(n)", MPS) for n=0:(nstates0-1)]
    energies0 = [read(fid, "E_$(n)") for n=0:(nstates0-1)]
    println("Energies before sorting: $(energies0)")
    sorting_inds = sortperm(energies0)
    energies0 = energies0[sorting_inds]
    println("Energies after sorting: $(energies0)")
    psis0 = psis0[sorting_inds]

    H = read(fid, "H", MPO)
    g = read(fid, "g")
    t = read(fid, "t")
    # "Short Circuit" evaluates as if isnothing(maxdim); maxdim=fid["maxdim"]
    isnothing(maxdim) && (maxdim = read(fid, "maxdim"))
    isnothing(energy_tol) && (energy_tol = read(fid, "energy_tol"))
    isnothing(kdim) && (kdim = read(fid, "kdim"))
    isnothing(nstates) && (nstates = read(fid, "nstates"))
    isnothing(nsweeps) && (nsweeps = read(fid, "nsweeps"))
    isnothing(cutoff) && (cutoff = read(fid, "cutoff"))
    close(fid)
    if isnothing(newname)
        println("No name given, using old file name!")
        newname = fname
    end
    printif("Old energies: $(energies0)", outputlevel)
    # This would make things much safer: prevents overwriting!
    # isfile(newname) && throw(DomainError(newname, "An existing file matches newname=($newname)"))    
    println("Initializing new file $(newname)")
    h5open(newname, "w") do fid
        fid["dims"] = dims
        fid["maxdim"] = maxdim
        fid["energy_tol"] = energy_tol
        fid["noise"] = noise
        fid["kdim"] = kdim
        fid["nstates"] = nstates
        fid["nsweeps"] = nsweeps
        fid["cutoff"] = cutoff
        fid["g"] = g
        fid["t"] = t
    end
    Lx, Ly, parity, bc = dims
    N = Lx*Ly
    inds = ind_array(Lx, Ly; simple=simple_inds)    
    sites = siteinds(psis0[1])
    # constructing guess MPS for new states
    state = ["0" for n=1:N]
    if parity == 1
        state[1] = "1"
    end
    psis_in = [random_mps(sites, state, linkdims=2) for n=1:nstates]
    psis_out = [MPS() for n=1:nstates]
    energies_out = zeros(Float64, nstates)

    for n=1:nstates0
        psis_in[n] = psis0[n]
        energies_out[n] = energies0[n]
    end

    observer = DMRGSizeObserver(energy_tol, force_gc)

    # TODO: reuse this syntax elsewhere. Maybe define function to further streamline?
    dmrg_kwargs = Dict(:nsweeps => nsweeps,
                        :cutoff => cutoff,
                        :observer => observer,
                        :outputlevel => outputlevel,
                        :noise => noise,
                        :eigsolve_krylovdim=>kdim)
    for n=1:nstates
        printif("n = $(n)", outputlevel)
        if n <= nstates0 && reconverge
            printif("Reconverging!", outputlevel)
            old_maxdim = maxlinkdim(psis_in[n])
            printif("Original state maximum dimension: $(old_maxdim)", outputlevel)
            this_maxdim = maxdim[maxdim .>=x old_maxdim]
            printif("Maxdims for $(n)th state: $(this_maxdim)", outputlevel)
            if n == 1
                energies_out[1], psis_out[1] = dmrg(H, psis_in[1]; maxdim=this_maxdim, dmrg_kwargs...)
            else
                _, psis_out[n] = dmrg(H, psis_out[1:n-1], psis_in[n]; 
                                weight=weight,
                                maxdim=this_maxdim,
                                re_kwargs...)
                energies_out[n] = real(inner(psis_out[n]', H, psis_out[n])) # in case orthogonality penalty applied
            end
            printif("E_new, E_old, E_new-E_old: ", outputlevel)
            printif("$(energies_out[n]), $(energies0[n]), $(energies_out[n]-energies0[n])", outputlevel)
        elseif n <= nstates0
            printif("Not reconverging!", outputlevel)
        else
            _, psis_out[n] = dmrg(H, psis_out[1:n-1], psis_in[n]; 
                            weight=weight, maxdim=maxdim,
                            dmrg_kwargs...)
            energies_out[n] = real(inner(psis_out[n]', H, psis_out[n]))
        end
        update_data(newname, "E_$(n-1)", energies_out[n])
        update_data(newname, "psi_$(n-1)", psis_out[n])
        printif("E_$(n-1): $(energies_out[n])", outputlevel)
        printif("Overlaps: ", outputlevel)
        for m = 1:n
            printif("<$(n-1)|$(m-1)>: $(abs(inner(psis_out[n]', psis_out[m])))", outputlevel)
        end
        printif("", outputlevel)
    end
    return energies_out, psis_out
end