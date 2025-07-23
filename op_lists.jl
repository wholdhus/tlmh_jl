function H0_OpSum(t, dims, inds; periodic=true, u=0, imp_inds=[])
    os = OpSum()
    Lx, Ly, parity, bc = dims
    Lx = Int(Lx)
    Ly = Int(Ly)
    parity = Int(parity)
	for ix=1:Lx
		for iy=1:Ly
			i0 = inds[ix, iy]
			os += t,"P",i0
			#
			pp_coeffs = [2im*t, t, t, t]
			pp_inds = [inds[mod1(ix+1,Lx), iy],
					   inds[mod1(ix-1,Lx), iy],
					   inds[mod1(ix+1,Lx), mod1(iy+1,Ly)],
					   inds[ix, mod1(iy+1,Ly)]]
			pp_ixs = [ix+1, ix-1, ix+1, ix]
			for i=1:length(pp_coeffs)
			    if periodic || 0 < pp_ixs[i] <= Lx 
			        s = bc_sign(bc, Lx, [pp_ixs[i]])
			        os += pp_coeffs[i]*s,"Cdag",i0,"Cdag",pp_inds[i]
			        os += conj(pp_coeffs[i])*s,"C",pp_inds[i],"C",i0
			    end
			end
			pm_coeffs = [-t,t,t]
			pm_inds = [inds[mod1(ix-1,Lx), iy],
					   inds[mod1(ix+1,Lx), mod1(iy+1,Ly)],
					   inds[ix, mod1(iy+1,Ly)]]
			pm_ixs = [ix-1, ix+1, ix]
			for i=1:length(pm_coeffs)
			    if periodic || 0 < pm_ixs[i] <= Lx 
			        s = bc_sign(bc, Lx, [pm_ixs[i]])
			        os += pm_coeffs[i]*s,"Cdag",i0,"C",pm_inds[i]
			        os += conj(pm_coeffs[i])*s,"Cdag",pm_inds[i],"C",i0
			    end
			end
		end
	end
	for i in imp_inds
		os += u,"P",i
	end
	return os
end


function add_A_summand(os, g; i, j)
	os += -g,"P",i,"P",j
	return os
end


function add_B_summand(os, g; i, j, k)
	os += g,"Cdag",i,"P",j,"C",k
	os += g,"Cdag",k,"P",j,"C",i
	os += -g,"Cdag",i,"P",j,"Cdag",k
	os += -g,"C",k,"P",j,"C",i
	return os
end


function add_C_summand(os, g; i, j, k, l)
	# ++++
	os += -g,"Cdag",i,"Cdag",j,"Cdag",k,"Cdag",l
	# +++-
	os += -g,"Cdag",i,"Cdag",j,"Cdag",k,"C",l
	os += g,"Cdag",i,"Cdag",j,"Cdag",l,"C",k
	os += g,"Cdag",i,"Cdag",k,"Cdag",l,"C",j
	os += -g,"Cdag",j,"Cdag",k,"Cdag",l,"C",i
	# ++--
	os += -g,"Cdag",i,"Cdag",j,"C",k,"C",l
	os += -g,"Cdag",i,"Cdag",k,"C",j,"C",l
	os += g,"Cdag",i,"Cdag",l,"C",j,"C",k
	os += g,"Cdag",j,"Cdag",k,"C",i,"C",l
	os += -g,"Cdag",j,"Cdag",l,"C",i,"C",k
	os += -g,"Cdag",k,"Cdag",l,"C",i,"C",j
	# +---
	os += g,"Cdag",i,"C",j,"C",k,"C",l
	os += -g,"Cdag",j,"C",i,"C",k,"C",l
	os += -g,"Cdag",k,"C",i,"C",j,"C",l
	os += g,"Cdag",l,"C",i,"C",j,"C",k
	# ----
	os += -g,"C",i,"C",j,"C",k,"C",l
	return os
end


function add_P1_sum(os, g, dims, inds; periodic=true)
    Lx, Ly, parity, bc = dims
    Lx = Int(Lx)
    Ly = Int(Ly)
    parity = Int(parity)
	for ix=1:Lx
		for iy=1:Ly
		    if periodic || ix-1 > 0 || ix-2 > 0
		        os = add_B_summand(os, g*bc_sign(bc, Lx, [ix, ix-2])*abs(bc_sign(bc, Lx, [ix-1])),
				            	   i=inds[ix,iy],
				               	   j=inds[mod1(ix-1,Lx), iy],
				            	   k=inds[mod1(ix-2,Lx), iy])
			end
			if periodic || ix-1 > 0
		        os = add_C_summand(os, g*bc_sign(bc, Lx, [ix, ix-1, ix-1, ix]),
				            	   i=inds[ix,iy],
				            	   j=inds[mod1(ix-1,Lx), iy],
				            	   k=inds[mod1(ix-1,Lx), mod1(iy+1,Ly)],
				            	   l=inds[ix,mod1(iy+1,Ly)])
		    end
		end
	end
	return os
end


function add_P2_sum(os, g, dims, inds; periodic=true)
    Lx, Ly, parity, bc = dims
    Lx = Int(Lx)
    Ly = Int(Ly)
    parity = Int(parity)
	for ix=1:Lx
		for iy=1:Ly
		    if periodic || ix-1 > 0
		        os = add_B_summand(os, -g*bc_sign(bc, Lx, [ix, ix-1]), #*abs(bc_sign(bc, Lx, [ix])),
				            	   i=inds[ix, iy],
				            	   j=inds[ix, mod1(iy-1,Ly)],
				            	   k=inds[mod1(ix-1,Lx), mod1(iy-1,Ly)])
		    end
		    if periodic || ix+1 <= Lx
		        os = add_B_summand(os, -g*bc_sign(bc, Lx, [ix+1, ix]), #*abs(bc_sign(bc, Lx, [ix])), 
				            	   i=inds[mod1(ix+1,Lx), iy],
				            	   j=inds[ix, iy],
				            	   k=inds[ix, mod1(iy-1,Ly)])
		    end
		end
	end
	return os
end


function add_P3_sum(os, g, dims, inds; periodic=true)
    Lx, Ly, parity, bc = dims
    Lx = Int(Lx)
    Ly = Int(Ly)
    parity = Int(parity)
	for ix=1:Lx
		for iy=1:Ly
		    if periodic || ix+1 <= Lx
		        os = add_A_summand(os, g*abs(bc_sign(bc, Lx, [ix, ix+1])),
		                           i=inds[ix, iy], 
		                           j=inds[mod1(ix+1,Lx), iy])
		    end
		    if periodic || ix+1 <= Lx || ix+2 <= Lx
		        os = add_C_summand(os, g*bc_sign(bc, Lx, [ix, ix+1, ix+2, ix+1]),
					               i=inds[ix,iy],
					               j=inds[mod1(ix+1,Lx), iy],
					               k=inds[mod1(ix+2,Lx), mod1(iy+1,Ly)],
					               l=inds[mod1(ix+1,Lx), mod1(iy+1,Ly)])
		    end
		end
	end
	return os
end

function add_impurity(os, u, i)
	os += u,"P",i
	return os
end


function HI_OpSum(g, dims, inds; periodic=true)
    os = OpSum()
    if dims[4] == 0.0
        # println("OBC!")
        periodic = false
    end
    os = add_P1_sum(os, g, dims, inds; periodic=periodic)
    os = add_P2_sum(os, g, dims, inds; periodic=periodic)
    os = add_P3_sum(os, g, dims, inds; periodic=periodic)
    return os
end    
    
function H_OpSum(t, g::Number, dims, inds; periodic=true, u=0, imp_inds=[])
    if dims[4] == 0.0
        # println("OBC!")
        periodic = false
    end
    os = H0_OpSum(t, dims, inds; periodic=periodic, 
				  u=u, imp_inds=imp_inds)
    os = add_P1_sum(os, g, dims, inds; periodic=periodic)
    os = add_P2_sum(os, g, dims, inds; periodic=periodic)
    os = add_P3_sum(os, g, dims, inds; periodic=periodic)
    return os
end

function H_OpSum_asym(t, g::Vector, dims, inds, periodic=true)
	println("Using g = $g")
     if dims[4] == 0.0
        # println("OBC!")
        periodic = false
    end
    os = H0_OpSum(t, dims, inds; periodic=periodic)
    os = add_P1_sum(os, g[1], dims, inds; periodic=periodic)
    os = add_P2_sum(os, g[2], dims, inds; periodic=periodic)
    os = add_P3_sum(os, g[3], dims, inds; periodic=periodic)
    return os
end

function P1_OpSum(t, g, dims, inds)
    os = OpSum()

	periodic = true
    if dims[4] == 0.0
        # println("OBC!")
        periodic = false
    end
    os = add_P1_sum(os, g, dims, inds, periodic=periodic)
    return os
end

function P2_OpSum(t, g, dims, inds)
    os = OpSum()

	periodic = true
    if dims[4] == 0.0
        # println("OBC!")
        periodic = false
    end
    os = add_P2_sum(os, g, dims, inds, periodic=periodic)
    return os
end

function P3_OpSum(t, g, dims, inds)
    os = OpSum()

	periodic = true
    if dims[4] == 0.0
        # println("OBC!")
        periodic = false
    end
    os = add_P3_sum(os, g, dims, inds, periodic=periodic)
    return os
end

function parity_break_OpSum(eps, dims)
    Lx, Ly, parity, bc = dims
	N = Lx*Ly
    os = OpSum()

	for i = 1:N
		os += eps,"C",i
		os += eps,"Cdag",i
	end
	return os
end