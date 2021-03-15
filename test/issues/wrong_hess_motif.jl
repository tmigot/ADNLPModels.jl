#=
Bug in the sparsity detection ?
I realized there was more similar issues:
https://github.com/SciML/SparsityDetection.jl/issues/42
so they will soon give up on this package?
I think we now have to go here:
https://mtk.sciml.ai/dev/highlevel/#Sparsity-Detection-1
=#

using SparsityDetection, SparseArrays

x0 = [1., 2.]
f(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2
fbis(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 + 1

s = hessian_sparsity(f, x0)
@show s == ones(Bool,2,2)

s = hessian_sparsity(fbis, x0)
@show s == spzeros(2,2)