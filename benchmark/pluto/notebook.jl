### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 32697e04-a37e-11eb-09a8-0d09a079ea91
begin
	using Images, ImageIO, PlutoUI
	using ADNLPModels
end

# ╔═╡ 18e736ec-701d-4fc0-ac9c-5bfe23500ebe
begin
	using NLPModels
end

# ╔═╡ ad18f3bf-8f04-471a-9a56-e3abf25389a4
md"
# What's (will be) new in ADNLPModels

In the last update of ADNLPModels, we can use different backend for the automatic differentiation ForwardDiff (by default), ReverseDiffAD, or ZygoteAD.
"

# ╔═╡ 7cfedb58-d6e5-4795-968e-2ec6bc3c038c
begin
	nlp = ADNLPModel(x->sum(x), ones(3), adbackend=ADNLPModels.ZygoteAD())
end

# ╔═╡ 55203dc8-a9bc-486d-ad28-786109f5dd26
begin
	grad(nlp, nlp.meta.x0)
end

# ╔═╡ 4a6641d5-28fe-4fdb-a197-e66c40ee010d
md"
This is great, and open the door to new improvements in fine-tuning the use of these new backends. In the current version, all the matrices (jacobian and hessian) are dense matrices.
"

# ╔═╡ 0361ef65-c215-4c2e-8bc4-0f9d01d41d3a
begin
	hess(nlp, nlp.meta.x0)
end

# ╔═╡ 33b210f3-bfb9-4dae-adb7-5f1b64862be0
md"
A couple of weeks ago, Alexis and I started studying an alternative `optimized` ADNLPModels with sparse derivatives. The work in progress are in the branches:
* [amontoison/ADNLPModels.jl#sparse-dev](https://github.com/amontoison/ADNLPModels.jl/tree/sparse-dev) -> [tmigot/ADNLPModels.jl#sparse-dev](https://github.com/tmigot/ADNLPModels.jl/tree/sparse-dev)
* [tmigot/ADNLPModels.jl#benchmark](https://github.com/tmigot/ADNLPModels.jl/tree/benchmark) (for today's benchmark and integration)
The aim is twofold:
* *Get the best of all the backends*. Compute each function with the best out of ForwardDiff, ReverseDiff, Zygote. For instance, we use ReverseDiff for the gradient.
* *Get sparse derivatives* for Hessian and Jacobian matrices.

These novelties rely heavily on comparing different backends, so we have a benchmark folder with 73 **scalable** examples including (only) 6 with scalable constraints (and 21 small size constrained examples from OptimizationProblems.jl).
"

# ╔═╡ 7ff17c72-92f0-45ac-bdbb-bf71d724d4c8
begin
	load("fig/2021-04-22_32_21_perf-grad.png")
end

# ╔═╡ 002bee66-3a9d-4575-810f-320546f34e9c
md"
## Sparse derivatives

In order to compute the sparsity structure and the hessian matrix, we are using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl). In greater details, we store some computation at the initialization of the ADNLPModel.
"

# ╔═╡ 5801583c-f189-4587-a96f-e983be8d1b6d


# ╔═╡ f56aa24d-cdc9-4512-914e-20d7aeda1fbc
md"
Then, we can use the precomputed function in hess_coord!.
"

# ╔═╡ 95463297-f6f5-4eef-9a01-662141b377bc


# ╔═╡ dc7db909-c76e-441e-a500-a16e89766bbe
begin
	load("fig/2021-04-22_32_21_perf-hess_coord.png")
end

# ╔═╡ 0f215736-7019-48e1-b6ee-0f83f7c3f685
begin
	load("fig/2021-04-22_32_21_perf-jac_coord.png")
end

# ╔═╡ e80a274e-6530-40db-9721-44228ffd1e2e
md"
## Upcoming/Questions
About the benchmark:
- Need more problems with ADNLPModels/JuMP implementation;
- Should we keep the benchmark in ADNLPModels.jl or elsewhere?
- Study `Jprod` and `Hprod`
- Improve the `hess_structure` and `jac_structure` calls
- Use Symbolics.jl to simplify the functions `f` and `c`

#### Migration to ADNLPModels.jl

- [ ] Update master branch of ADNLPModels to handle sparse structure (while dense is assumed now)
- [ ] Make a SymbolicsAD with Symbolics as an optional package.
- [ ] Make an DiffAD when all optional packages are available with the best of each.
- [ ] Get the benchmark running after every PR?
"

# ╔═╡ Cell order:
# ╠═32697e04-a37e-11eb-09a8-0d09a079ea91
# ╠═18e736ec-701d-4fc0-ac9c-5bfe23500ebe
# ╟─ad18f3bf-8f04-471a-9a56-e3abf25389a4
# ╠═7cfedb58-d6e5-4795-968e-2ec6bc3c038c
# ╠═55203dc8-a9bc-486d-ad28-786109f5dd26
# ╟─4a6641d5-28fe-4fdb-a197-e66c40ee010d
# ╠═0361ef65-c215-4c2e-8bc4-0f9d01d41d3a
# ╟─33b210f3-bfb9-4dae-adb7-5f1b64862be0
# ╟─7ff17c72-92f0-45ac-bdbb-bf71d724d4c8
# ╟─002bee66-3a9d-4575-810f-320546f34e9c
# ╠═5801583c-f189-4587-a96f-e983be8d1b6d
# ╟─f56aa24d-cdc9-4512-914e-20d7aeda1fbc
# ╠═95463297-f6f5-4eef-9a01-662141b377bc
# ╠═dc7db909-c76e-441e-a500-a16e89766bbe
# ╠═0f215736-7019-48e1-b6ee-0f83f7c3f685
# ╟─e80a274e-6530-40db-9721-44228ffd1e2e
