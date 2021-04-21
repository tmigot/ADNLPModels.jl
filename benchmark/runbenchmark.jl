using Pkg
Pkg.activate("")
using BenchmarkTools, DataFrames, JuMP, Plots, Profile
#using ProfileView
#JSO packages
using NLPModels, BenchmarkProfiles, SolverBenchmark, OptimizationProblems, NLPModelsJuMP
#This package
using ADNLPModels, ReverseDiff, Zygote, ForwardDiff
#Render
using JLD2, Dates

#problems = ["hs5", "brownden"]
problems2 = ["arglina", "arglinb", "arglinc", "arwhead", "bdqrtic", "beale", "broydn7d",
             "brybnd", "chainwoo", "chnrosnb_mod", "cosine", "cragglvy", "curly10", "curly20", 
             "curly30", "dixon3dq", "dqdrtic",
             "dqrtic", "edensch", "eg2", "engval1", "errinros_mod", "extrosnb", "fletcbv2",
             "fletcbv3_mod", "fletchcr", "freuroth", "genhumps", "genrose", "genrose_nash",
             "indef_mod", "liarwhd", "morebv", "ncb20", "ncb20b", "noncvxu2", "noncvxun",
             "nondia", "nondquar", "NZF1", "penalty2", "penalty3", "powellsg", "power",
             "quartc", "sbrybnd", "schmvett", "scosine", "sparsine", "sparsqur", "srosenbr",
             "sinquad", "tointgss", "tquartic", "tridia", "vardim", "woods"]
# problems with constraints (none are scalable)
problems3 = ["hs6", "hs7", "hs8", "hs9", "hs26", "hs27", "hs28", "hs39", "hs40", "hs42", "hs46",
             "hs47", "hs48", "hs49", "hs50", "hs51", "hs52", "hs56", "hs63", "hs77", "hs79"]
#scalable constrained problems
problems4 = ["clnlbeam", "controlinvestment", "hovercraft1d", "polygon1", "polygon2", "polygon3"]
using JuMP

problems = problems3 #union(problems, problems2, problems3)

#List of problems used in tests
#Problems from NLPModels
#include("../test/problems/hs5.jl") #bounds constraints n=2, dense hessian
#include("../test/problems/brownden.jl") #unconstrained n=4, dense hessian

for pb in problems
  include("problems/$(lowercase(pb)).jl")
end

include("additional_func.jl")

#Extend the functions of each problems to the variants of RADNLPModel
nvar = 32 #targeted size >=31 /// doesn't really work because of OptimizationProblems
#=
for pb in problems #readdir("test/problems")
  eval(Meta.parse("$(pb)_radnlp_reverse(args... ; kwargs...) = $(pb)_radnlp(args... ; gradient = ADNLPModels.reverse, kwargs...)"))
  eval(Meta.parse("$(pb)_radnlp_smartreverse(args... ; kwargs...) = $(pb)_radnlp(args... ; gradient = ADNLPModels.smart_reverse, kwargs...)"))
  eval(Meta.parse("$(pb)_jump(args... ; kwargs...) = MathOptNLPModel($(pb)())"))
end
=#
for pb in problems #readdir("test/problems")
  eval(Meta.parse("$(pb)_radnlp_smartreverse(args... ; kwargs...) = $(pb)_radnlp(args... ; n=$(nvar), gradient = ADNLPModels.smart_reverse, kwargs...)"))
  eval(Meta.parse("$(pb)_reverse(args... ; kwargs...) = $(pb)_autodiff(args... ; adbackend=ADNLPModels.ReverseDiffAD(), n=$(nvar), kwargs...)"))
  eval(Meta.parse("$(pb)_zygote(args... ; kwargs...) = $(pb)_autodiff(args... ;adbackend=ADNLPModels.ZygoteAD(), n=$(nvar), kwargs...)"))
  eval(Meta.parse("$(pb)_jump(args... ; kwargs...) = MathOptNLPModel($(pb)($(nvar)))"))
end

models = [:reverse, :zygote, :autodiff, :radnlp_smartreverse, :jump] #[:radnlp_reverse, :radnlp_smartreverse, :autodiff]
fun    = Dict(:obj => (nlp, x) -> obj(nlp, x), 
              :grad => (nlp, x) -> grad(nlp, x),
              :hess => (nlp, x) -> hess(nlp, x), 
              :hess_coord => (nlp, x) -> hess_coord(nlp, x), 
              :hess_structure => (nlp, x) -> hess_structure(nlp),
              :jac => (nlp, x) -> (nlp.meta.ncon > 0 ? jac(nlp, x) : zero(eltype(x))),
              :jac_coord => (nlp, x) -> (nlp.meta.ncon > 0 ? jac_coord(nlp, x) : zero(eltype(x))),
              :jac_structure => (nlp, x) -> (nlp.meta.ncon > 0 ? jac_structure(nlp) : zero(eltype(x)))
              )
funsym = keys(fun)

rb = runbenchmark(problems, models, fun)
N = length(rb[first(funsym)][models[1]]) #number of problems x number of x
gstats = group_stats(rb, N, fun, models)

@save "$(today())_$(nvar)_bench_adnlpmodels.jld2" gstats

for f in funsym
  cost(df) = df.mean_time
  p = performance_profile(gstats[f], cost)
  png("$(today())_$(nvar)_perf-$(f)")
end
