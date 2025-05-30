function ind_array(Lx, Ly; simple=false)
	inds = zeros(Int, Lx, Ly)
	for ix=1:Lx
		for iy=1:Ly
			if simple
				inds[ix, iy] = Ly*(ix-1) + iy
			else
				if isodd(ix)
					inds[ix, iy] = Ly*ix + iy - Ly
				else
					inds[ix, iy] = Ly*ix - iy + 1
				end
			end
		end
	end
	return inds
end


function bc_sign(bc, Lx, ixs)
    factor = 1
	for ix in ixs
		if ix > Lx || ix < 1 
			factor *= bc
		end
	end
	return factor
end


function printif(msg, ol)
    if ol > 0
        println(msg)
    end
end
