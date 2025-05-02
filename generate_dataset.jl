using LinearAlgebra
using Random
using Statistics
using .Threads
using Plots
using FileIO
using Printf
using GeometryBasics
using JLD2

#global variables
omega = 1.0;
fluid_density = 1.0;
nx = 64;
ny = 32;	
deltaU=1e-7;
progress = 0;

function polygon_setboundary(nx, ny, n, xshift, yshift, radius)
    # Polygon center
    cx, cy = nx / 2 + xshift, ny / 2 + yshift

    # Generate sorted random angles and create convex polygon
    angles = sort(rand(n) .* 2π)
    points = [(cx + radius * cos(θ), cy + radius * sin(θ)) for θ in angles]

    # Define 2D cross product (scalar)
    cross2D(u::Tuple{<:Real,<:Real}, v::Tuple{<:Real,<:Real}) = u[1]*v[2] - u[2]*v[1]

    # Winding number point-in-polygon test
    function point_in_polygon(x, y, poly)
        wn = 0
        for i in 1:length(poly)
            x0, y0 = poly[i]
            x1, y1 = poly[mod1(i+1, length(poly))]
            if y0 <= y
                if y1 > y && cross2D((x1 - x0, y1 - y0), (x - x0, y - y0)) > 0
                    wn += 1
                end
            else
                if y1 <= y && cross2D((x1 - x0, y1 - y0), (x - x0, y - y0)) < 0
                    wn -= 1
                end
            end
        end
        return wn != 0
    end

    # Fill grid: 1 inside polygon, 0 outside
    BOUND = [point_in_polygon(i, j, points) ? 1 : 0 for i in 1:nx, j in 1:ny]

    ON = findall(x -> x > 0, vec(BOUND))
    OFF = findall(x -> x == 0, vec(BOUND))
    numactivenodes = length(OFF)

    return (BOUND, ON, OFF, numactivenodes)
end

function get_images(xshift, yshift; radius=7, n=4)
		#generates an input/output pair of images with a circle with the given position shift
		
		avu=1; prevavu=1;
		ts=0;
		avus = zeros(Float64, 100000)
		
		# single-particle distribution function
		# particle densities conditioned on one of the 9 possible velocities
		F = repeat([fluid_density/9], outer= [nx, ny, 9]);
		FEQ = F;
		msize = nx*ny;
		CI = collect(0:msize:msize*7); # offsets of different directions e_a in the F matrix 
		    
		#*
		# constants
		#*
		t1 = 4/9;
		t2 = 1/9;
		t3 = 1/36;
		c_squ = 1/3;

		#initialize arrays
		DENSITY = zeros(Int, nx, ny)
		UX = zeros(Int, nx, ny)
		UY = zeros(Int, nx, ny)
		
		#code that makes the blockage
		BOUND, ON, OFF, numactivenodes = polygon_setboundary(nx, ny, n, xshift, yshift, radius)
		
	
		# linear indices in F of occupied nodes
		TO_REFLECT=[ON.+CI[1] ON.+CI[2] ON.+CI[3] ON.+CI[4] ON.+CI[5] ON.+CI[6] ON.+CI[7] ON.+CI[8]];
		
		# Right <-> Left: 1 <-> 5; Up <-> Down: 3 <-> 7
		#(1,1) <-> (-1,-1): 2 <-> 6; (1,-1) <-> (-1,1): 4 <-> 8
		REFLECTED= [ON.+CI[5] ON.+CI[6] ON.+CI[7] ON.+CI[8] ON.+CI[1] ON.+CI[2] ON.+CI[3] ON.+CI[4]];
	
		#it errors without this, probably not efficient
		while ((ts<100000) & (1e-10 < abs((prevavu-avu)/avu))) | (ts<100)
		    #*
		    # Streaming
			# maybe use the inline if else statements for the third index and parallelize
		    #*
		    # particles at (x,y)=[2,1] were at [1, 1] before: 1 points right (1,0)
		    F[:,:,1] = vcat(F[nx, :, 1]',F[1:nx-1,:,1]);
		    
			# particles at [1,2] were at [1, 1] before: 3 points up (0,1)
			F[:,:,3] = hcat(F[:, ny, 3], F[:, 1:ny-1,3]);
	
		    # particles at [1,1] were at [2, 1] before: 5 points left (-1,0)
			F[:,:,5] = vcat(F[2:nx,:,5], F[1,:,5]');
		    
			# particles at [1,1] were at [1, 2] before: 7 points down (0,-1)
			F[:,:,7] = hcat(F[:, 2:ny, 7], F[:, 1, 7]);
			
		    # particles at [2,2] were at [1, 1] before: 2 points to (1,1)
			F[:,:,2] = F[cat([nx], collect(1:nx-1), dims=1),cat([ny], collect(1:ny-1), dims=1),2];
	
			
		    # particles at [1,2] were at [2, 1] before: 4 points to (-1, 1)
		    F[:,:,4] = F[cat(collect(2:nx), [1], dims=1),cat([ny], collect(1:ny-1), dims=1),4];
		    # particles at [1,1] were at [2, 2] before: 6 points to (-1, -1)
		    F[:,:,6] = F[cat(collect(2:nx), [1], dims=1),cat(collect(2:ny), [1], dims=1),6];
		    # particles at [2,1] were at [1, 2] before: 8 points down (1,-1)
		    F[:,:,8] = F[cat([nx], collect(1:nx-1), dims=1),cat(collect(2:ny), [1], dims=1),8];
		    
		    DENSITY = sum(F,dims=3);
		    # 1,2,8 are moving to the right, 4,5,6 to the left
		    # 3, 7 and 9 don't move in the x direction
		    UX = (sum(F[:,:,[1, 2, 8]],dims=3)-sum(F[:,:,[4, 5, 6]],dims=3))./DENSITY;
		    # 2,3,4 are moving up, 6,7,8 down
		    # 1, 5 and 9 don't move in the y direction
		    UY = (sum(F[:,:,[2, 3, 4]],dims=3)-sum(F[:,:,[6, 7, 8]],dims=3))./DENSITY;
		    
		    UX[1,1:ny] = UX[1,1:ny] .+ deltaU; #Increase inlet pressure
		    
		    UX[ON] .= 0; UY[ON] .= 0; DENSITY[ON] .= 0;
		    
		    U_SQU = UX.^2+UY.^2;
		    U_C2 = UX+UY;
		    U_C4 = -UX+UY;
		    U_C6 = -U_C2;
		    U_C8 = -U_C4;
		    
		    # Calculate equilibrium distribution: stationary (a = 0)
		    FEQ[:,:,9] = t1*DENSITY.*(1 .-U_SQU/(2*c_squ));
		    
		    # nearest-neighbours
		    FEQ[:,:,1] = t2*DENSITY.*(1 .+UX/c_squ.+0.5*(UX/c_squ).^2 .-U_SQU/(2*c_squ));
		    FEQ[:,:,3] = t2*DENSITY.*(1 .+UY/c_squ.+0.5*(UY/c_squ).^2 .-U_SQU/(2*c_squ));
		    FEQ[:,:,5] = t2*DENSITY.*(1 .-UX/c_squ.+0.5*(UX/c_squ).^2 .-U_SQU/(2*c_squ));
		    FEQ[:,:,7] = t2*DENSITY.*(1 .-UY/c_squ.+0.5*(UY/c_squ).^2 .-U_SQU/(2*c_squ));
		    
		    # next-nearest neighbours
		    FEQ[:,:,2] = t3*DENSITY.*(1 .+U_C2/c_squ.+0.5*(U_C2/c_squ).^2 .-U_SQU/(2*c_squ));
		    FEQ[:,:,4] = t3*DENSITY.*(1 .+U_C4/c_squ.+0.5*(U_C4/c_squ).^2 .-U_SQU/(2*c_squ));
		    FEQ[:,:,6] = t3*DENSITY.*(1 .+U_C6/c_squ.+0.5*(U_C6/c_squ).^2 .-U_SQU/(2*c_squ));
		    FEQ[:,:,8] = t3*DENSITY.*(1 .+U_C8/c_squ.+0.5*(U_C8/c_squ).^2 .-U_SQU/(2*c_squ));
		    
		    BOUNCEDBACK = F[TO_REFLECT]; #Densities bouncing back at next timestep
		    
		    F = omega*FEQ + (1-omega)*F;
		    
		    F[REFLECTED] = BOUNCEDBACK;
		    prevavu = avu;
		    avu = sum(sum(UX))/numactivenodes;
		    avus[ts+1] = prevavu-avu;
		    ts = ts+1;
		end
		
		return [BOUND, UX, UY, DENSITY]
	end

function plot_everything(pltBOUND, pltUX, pltUY, pltDENSITY, number)
	# Flip BOUND as imshow in PyPlot uses origin="lower"
		pltOFF = findall(x -> x == 0, vec(pltBOUND))
		data = 1 .- pltBOUND'
		
		# Plot heatmap
		a = heatmap(
		    data,
		    c=:hot,
		    clims=(0.0, 1.0),
		    aspect_ratio=:equal,
		    xlabel="x",
		    ylabel="y",
		    title="Input Image",
		    framestyle=:box,
			colorbar=false
		)
		
		# Create folder and save image
		savepath = "/home/p170x-1/Documents/InputImages/"
		mkpath(savepath)
		filename = string(savepath, "input", number, ".png")
		savefig(a, filename)

		# Correct 2D array transposition using permutedims
		base_img = 1 .- pltBOUND'
		density_img = pltDENSITY[:,:]'
		density_min = mean(pltDENSITY[pltOFF]) - 3std(pltDENSITY[pltOFF])
		
		# Plot base image (hot background)
		b= heatmap(
		    base_img,
		    color=:hot,
		    clims=(0.0, 1.0),
		    aspect_ratio=:equal,
		    xlabel="x",
		    ylabel="y",
		    framestyle=:box,
		    legend=false
		)
		
		# Overlay density (transparent bwr layer)
		heatmap!(
		    density_img,
		    color=:bwr,
		    clims=(density_min, maximum(pltDENSITY)),
		    alpha=0.6,
		    legend=true
		)
		
		# Transpose velocity fields for quiver plot
		UXp = pltUX[:, :]'
		UYp = pltUY[:, :]'
		
		X = repeat(collect(1:nx)', ny, 1)
		Y = repeat(collect(1:ny), 1, nx)

		quiver!(
		    vec(X),
		    vec(Y),
		    quiver=(vec(UXp), vec(UYp)),
		    color=:black,
		    lw=0.5,
		    legend=false
		)
	
		# Title with math formatting
		title!("Converged Flow Field")
		
		# Set y-limits
		ylims!(0, ny)
		xlims!(0, nx)
		
		# Save to file
		savepath = "/home/p170x-1/Documents/OutputImages/"
		mkpath(savepath)
		filename = string(savepath, "output", number, ".png")
		savefig(b, filename)
end

# Thread-safe dataset generation
function create_dataset(n_samples::Int, nx::Int=64, ny::Int=32, savefile::String="fluid_data.jld2")
    nthreads = Threads.nthreads()
    println("Using $nthreads threads.")

    # Preallocate storage
    xdata = Array{Float32}(undef, nx, ny, 1, n_samples)
    ydata = Array{Float32}(undef, nx, ny, 3, n_samples)

    Threads.@threads for i in 1:n_samples
    	if progress % 100 == 0
    		println(i)
	end
	
        # Ensure thread-local RNG for reproducibility (I didn't do this)
		radius = rand(5:12)
		xshift = rand(-20:20)
		yshift = rand(-5:5)
		n = rand(3:8)
		
        BOUND, ux, uy, rho = get_images(xshift, yshift; radius=radius, n=n)

        # Safely write to preallocated slot
        xdata[:, :, 1, i] = BOUND
        ydata[:, :, 1, i] = ux
        ydata[:, :, 2, i] = uy
        ydata[:, :, 3, i] = rho
        progress += 1
    end

    # Save dataset
    @save savefile xdata ydata
    println("Saved dataset with $n_samples samples to $savefile")
end

create_dataset(10000) #make the dataset

@load "fluid_data.jld2" xdata ydata #loads the file into arrays

for _ in 1:10
	n = rand(1:10000)
	BOUND = xdata[:, :, 1, n]
	Ux = ydata[:, :, 1, n]
	Uy = ydata[:, :, 2, n]
	rho = ydata[:, :, 3, n]

	plot_everything(BOUND, Ux, Uy, rho, n)
end
