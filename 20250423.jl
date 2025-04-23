### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 9035c031-7a52-4f1f-ae22-1e5a9a02500c
begin
	using LinearAlgebra
	using Random
	using Statistics
	using PyPlot
	using .Threads
end

# ╔═╡ dfd8a12b-3517-4747-a052-09399fe245d7
md"""
0. What do we want the images to look like? Good as is?
1. Generalizable setboundary function (maybe done?)
2. Generate sufficient training data
3. CNN structure + training
4. Output images
"""

# ╔═╡ 7629d59a-c892-4c38-8c6a-5164b313c451
md"""
todo:
1. plot object as black
2. plot convergence 
3. random circle generator (done. We can input shifts and get just the circle and the steady state flow as two separate images)

"""

# ╔═╡ 2172e7f7-3f09-4ed0-885b-48f4812dc7f9
function circle_setboundary(nx, ny, radius, xshift, yshift)
	    #BOUND = rand(nx,ny) .> 1 - fraction; # random domain
	    center = [nx/2 + xshift, ny/2 + yshift]
	    BOUND = [norm([i, j]-center)  < radius ? 1 : 0 for i=1:nx, j=1:ny]
		
		ON = findall(x -> x >0, vec(BOUND)); # matrix offset of each Occupied Node
	    OFF = findall(x-> x<1, vec(BOUND));
	
		numactivenodes= sum(sum(-1 .* BOUND .+ 1));

	return (BOUND, ON, OFF, numactivenodes)
	end

# ╔═╡ e082309c-e705-4c22-8e27-482930120c5f
#shamelessly used chatgpt for this
function random_boundary(nx, ny, num_vertices)
    # Generate a random set of vertices within the grid
    vertices = [(rand(1:nx), rand(1:ny)) for _ in 1:num_vertices]
    
    # Sort vertices in a counter-clockwise order to form a polygon
    function angle_from_center(vertex)
        cx, cy = nx/2, ny/2
        return atan(vertex[2] - cy, vertex[1] - cx)
    end
	
    sorted_vertices = sort(vertices, by=angle_from_center)
    
    # Create the polygon boundary (this uses the ray-casting algorithm to check if a point is inside)
    BOUND = zeros(Int, nx, ny)
    
    # A helper function to check if a point is inside the polygon
    function point_in_polygon(x, y, vertices)
        inside = false
        n = length(vertices)
        p1x, p1y = vertices[1]
        for i in 1:n+1
            p2x, p2y = vertices[mod(i, n)+1]
            if y > min(p1y, p2y)
                if y <= max(p1y, p2y)
                    if x <= max(p1x, p2x)
                        if p1y != p2y
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        end
                        if p1x == p2x || x <= xinters
                            inside = !inside
                        end
                    end
                end
            end
            p1x, p1y = p2x, p2y
        end
        return inside
    end
    
    # Fill the BOUND array based on whether each point is inside the polygon
    for i in 1:nx
        for j in 1:ny
            if point_in_polygon(i, j, sorted_vertices)
                BOUND[i, j] = 1
            end
        end
    end
    
    # Get the indices of the occupied nodes (ON)
    ON = findall(x -> x > 0, vec(BOUND))
    OFF = findall(x-> x<1, vec(BOUND));

    # Count the number of active nodes
    numactivenodes = sum(BOUND)
    
    return BOUND, ON, OFF, numactivenodes
end


# ╔═╡ 8c063ce1-3095-450f-946f-1c8dfe87c6f2
begin
	#define constants
	omega = 1.0;
	density = 1.0;
	radius = 5.
	
	nx = 64;
	ny = 32;	

	deltaU=1e-7;


	function get_images(xshift, yshift, save=true, convergencePlot=false)
		#generates an input/output pair of images with a circle with the given position shift

		avu=1; prevavu=1;
		ts=0;
		global avus = zeros(Float64, 100000)
		
		# single-particle distribution function
		# particle densities conditioned on one of the 9 possible velocities
		F = repeat([density/9], outer= [nx, ny, 9]);
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
		 
		
		#code that makes the blockage
		BOUND, ON, OFF, numactivenodes = circle_setboundary(nx, ny, 5, xshift, yshift)
		#return BOUND/ON separately to make the train/test dataset?
	
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
		    
		    global DENSITY = sum(F,dims=3);
		    # 1,2,8 are moving to the right, 4,5,6 to the left
		    # 3, 7 and 9 don't move in the x direction
		    global UX = (sum(F[:,:,[1, 2, 8]],dims=3)-sum(F[:,:,[4, 5, 6]],dims=3))./DENSITY;
		    # 2,3,4 are moving up, 6,7,8 down
		    # 1, 5 and 9 don't move in the y direction
		    global UY = (sum(F[:,:,[2, 3, 4]],dims=3)-sum(F[:,:,[6, 7, 8]],dims=3))./DENSITY;
		    
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
		if save==true
			figure();
			imshow(1 .-BOUND', cmap="hot", interpolation="None", vmin=0., vmax=1., origin="lower");
			title(string("Circle of radius ", string(radius), " at (", string(round(nx/2 + xshift)), ",", string(round(ny/2 + yshift)), ")"));
			xlabel("x");
			ylabel("y");
			# mkpath("/Users/charlie/Documents/InputImages/")
			# savefig(string("/Users/charlie/Documents/InputImages/input", string(xshift), ".", string(yshift), ".jpg"), dpi=1200)
			mkpath("/Users/evrydayorsted/Documents/harddrive/classes/compPhys/fluids/InputImages/")
			savefig(string("/Users/evrydayorsted/Documents/harddrive/classes/compPhys/fluids/InputImages/input", string(xshift), ".", string(yshift), ".jpg"), dpi=1200)
	
			figure();
			imshow(1 .-BOUND', cmap="hot", interpolation="None", vmin=0., vmax=1., origin="lower");
			imshow(DENSITY[:,:]', cmap = "bwr",vmin=mean(DENSITY[OFF])- 3std(DENSITY[OFF]))
	
			quiver(1:nx-1, 0:ny-1, UX[2:nx,:]', UY[2:nx,:]');
			title(string("Flow field after \$ ", string(ts), " \\delta t\$"));
			xlabel("x");
			ylabel("y");
			ylim(0,ny);
			# mkpath("/Users/charlie/Documents/OutputImages")
			# savefig(string("/Users/charlie/Documents/OutputImages/output", string(xshift), ".", string(yshift), ".jpg"), dpi=1200)
			mkpath("/Users/evrydayorsted/Documents/harddrive/classes/compPhys/fluids/OutputImages/")
			savefig(string("/Users/evrydayorsted/Documents/harddrive/classes/compPhys/fluids/OutputImages/output", string(xshift), ".", string(yshift), ".jpg"), dpi=1200)
			close("all")
		end
		if convergencePlot==true
			figure();
			plot(avus[2:length(avus)]);
			title("avus convergence");
			xlabel("x");
			ylabel("y");
			show()
		end
		end
end

# ╔═╡ f24b51d8-fc98-4f60-8223-0f5c0dc87d19
@time for i in -1:1
	for j in -1:1
		get_images(i, j)
	end
end

# ╔═╡ 06c7970f-a49f-4ec9-8bf2-44112b5bc860
@time for i in -1:1
	@threads for j in -1:1
		get_images(i, j)
	end
end

# ╔═╡ eb2a643b-3ada-463b-9f4b-54db56e2bddb


# ╔═╡ 0e2d2b2d-5f74-437a-aedf-ee9eb3d9b579


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
PyPlot = "~2.11.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "4bb2dc17a2b134a96b8b03161d77a0d2cd76ca12"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "c7acce7a7e1078a20a285211dd73cd3941a871d6"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.0"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "b19db3927f0db4151cb86d073689f2428e524576"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.2"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+2"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "d2c2b8627bbada1ba00af2951946fb8ce6012c05"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.11.6"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"
"""

# ╔═╡ Cell order:
# ╠═9035c031-7a52-4f1f-ae22-1e5a9a02500c
# ╟─dfd8a12b-3517-4747-a052-09399fe245d7
# ╟─7629d59a-c892-4c38-8c6a-5164b313c451
# ╠═2172e7f7-3f09-4ed0-885b-48f4812dc7f9
# ╠═e082309c-e705-4c22-8e27-482930120c5f
# ╠═8c063ce1-3095-450f-946f-1c8dfe87c6f2
# ╠═f24b51d8-fc98-4f60-8223-0f5c0dc87d19
# ╠═06c7970f-a49f-4ec9-8bf2-44112b5bc860
# ╠═eb2a643b-3ada-463b-9f4b-54db56e2bddb
# ╠═0e2d2b2d-5f74-437a-aedf-ee9eb3d9b579
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
