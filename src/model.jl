export RADNLPModel

"""
    RADNLPModel(f, x0)
    RADNLPModel(f, x0, lvar, uvar)
RADNLPModel is an AbstractNLPModel using automatic differentiation to compute the derivatives.
The problem is defined as
     min  f(x)
    s.to  lcon ≤ c(x) ≤ ucon
          lvar ≤   x  ≤ uvar.
The following keyword arguments are available to all constructors:
- `name`: The name of the model (default: "Generic")
The following keyword arguments are available to the constructors for constrained problems:
- `lin`: An array of indexes of the linear constraints (default: `Int[]`)
- `y0`: An inital estimate to the Lagrangian multipliers (default: zeros)
"""
mutable struct RADNLPModel <: AbstractNLPModel #AbstractADNLPModel
  meta :: NLPModelMeta
  counters :: Counters
  # Functions
  f :: Function
  c :: Function
  ∇f! :: Function
  cfg #config gradient Union{Nothing, ReverseDiff.CompiledTape}

  cfH #config for the Hessian computation
  cfJ :: Union{ForwardColorJacCache, Nothing} #config for the Jacobian computation
end

show_header(io :: IO, model :: RADNLPModel) = println(io, "RADNLPModel - Model with automatic differentiation")

"""
Create an RADNLPModel with ReverseDiff.gradient! and precompilation work.
"""
function smart_reverse(x0 :: AbstractVector,
                       f  :: Function)

  # build in-place objective gradient
  v = similar(x0)
  #global k = -1; filler() = (k = -k; k); fill!(v, filler())
  f_tape = ReverseDiff.GradientTape(f, v)
  cfg = ReverseDiff.compile(f_tape) #compiled_f_tape typeof(compiled_f_tape) <: ReverseDiff.CompiledTape
  ∇f!(g, x, cfg) = ReverseDiff.gradient!(g, cfg, x)

  return (∇f!, cfg)
end

"""
Create an RADNLPModel with ReverseDiff.gradient!
"""
function reverse(:: AbstractVector,
                 f :: Function)

  cfg = nothing
  ∇f!(g, x, cfg) = ReverseDiff.gradient!(g, f, x)

return (∇f!, cfg)
end

#=
function RADNLPModel(meta :: AbstractNLPModelMeta,
                     f    :: Function,
                     cfH  :: ForwardColorJacCache;
                     gradient :: Function = smart_reverse)
  
  counters = Counters()
  #=
  cfg = nothing
  function ∇f!(g, x, cfg)
    _g = Zygote.gradient(f, x)
    g .= typeof(_g) <: AbstractVector ? _g : _g[1] #see benchmark/bug_zygote.jl
    return g
  end
  =#
  (∇f!, cfg) = gradient(meta.x0, f)
  # build v → ∇²f(x)v
  # NB: none of the options below works with compiled tapes
  function ∇²fprod!(x::AbstractVector, v::AbstractVector, Hv::AbstractVector)
    z = map(ForwardDiff.Dual, x, v)  # x + ε * v
    ∇fz = similar(z)
    ∇f!(∇fz, z, nothing)                      # ∇f(x + ε * v) = ∇f(x) + ε * ∇²f(x)v
    Hv = ForwardDiff.extract_derivative!(Nothing, Hv, ∇fz)  # ∇²f(x)v
    return Hv
  end
  # compute Hessian-vector product ∇²f(x)v by differentiating ϕ(x) := ∇f(x)ᵀv
  # this works but allocates a vector at each eval of ϕ(x)
  # ∇²fprod!(x, v, hv) = ReverseDiff.gradient!(hv,
  #                                         x -> begin
  #                                           g = similar(x)
  #                                           return dot(∇f!(g, x), v)
  #                                         end,
  #                                         x)
  # ∇²fprod!(x, v, hv) = begin
  #   g = ReverseDiff.track.(zero(x))
  #   ReverseDiff.gradient!(hv, x -> dot(∇f!(g, x), v), x)
  # end
  # function jacvec!(model::RADNLPModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
  #   z = map(ForwardDiff.Dual, x, v)  # z = x + ε * v
  #   fz = f(z)  # f(x + εv) = f(x) + ε * J(x) * v
  #   jv = ForwardDiff.extract_derivative!(Nothing, jv, fz)  # Jf(x) * y
  #   return Ap
  # end
  return RADNLPModel(meta, counters, f, x->T[], ∇f!, cfg, cfH) #, ∇²fprod!)
end
=#

#
# Tangi:
# On peut éviter le dernier appel à `sparse`, qui construie la matrice creuse,
# en donannt le mot clé `sparsity`
# https://github.com/SciML/SparsityDetection.jl/blob/74423959527e624b0ba7388b2c7de8d2039bcfc7/src/jacobian.jl#L125
#
#  Le code suivant donnerait un résultat différent, car il donne "juste" les non-zeros de la matrice.
#  J = jacobian_sparsity(model.c, y, model.meta.x0)
#  rows, cols, _ = findnz(J)
#
# jacobian_sparsity(f,output,input, sparsity = s, raw = true, verbose = false)
# is faster than 
# jacobian_sparsity(f,output,input, sparsity = s, verbose = false)
#
# Float64.(sparse(s))
# est plus rapide que
# sparse(s.I, s.J, 0., s.m, s.n)
#
#Issue with @code_warntype: the type of cfJ cannot be completely inferred
"""
Given an in-place function `f(dx, x)` of size `m`, and a vector `x0` of size `n`,
returns the number of non-zeros in the jacobian (after forming the sparse matrix),
and the config data.
"""
function _meta_function(f :: Function, x0 :: AbstractVector{T}, m :: Int, n :: Int) where T
  
  #We run (almost) the whole procedure once to get the non-zeros and the config
  output = similar(x0)
  s = Sparsity(m, n)
  jacobian_sparsity(f, output, x0, sparsity = s, raw = true, verbose = false)
  S = T.(sparse(s))
  colors = matrix_colors(S)
  cfJ = ForwardColorJacCache(f, x0, colorvec = colors, sparsity = S)
  nnzh = nnz(S) 

 return nnzh, cfJ
end

function _meta_function(f :: Function, x0 :: AbstractVector{T}, n :: Int) where T
  
  #We run (almost) the whole procedure once to get the non-zeros and the config
  #`hessian_sparsity` handles the whole matrix and not just the lower triangular
  s = hessian_sparsity(f, x0) #annoying print here!
  S = T.(tril(s))
  if s.nzval != []
    colors = matrix_colors(S)
    cfJ = ForwardColorJacCache(f, x0, colorvec = colors, sparsity = S)
  else
    cfJ = nothing #do better here?
  end

  nnzh = nnz(S) 

 return nnzh, cfJ
end

function RADNLPModel(f        :: Function, 
                     x0       :: AbstractVector{T}; 
                     name     :: String="GenericADNLPModel", 
                     gradient :: Function = smart_reverse, 
                     kwargs...) where T
  nvar = length(x0)
  @lencheck nvar x0

  (∇f!, cfg) = gradient(x0, f)
  nnzh, cfH = _meta_function(f, x0, nvar)
  
  meta = NLPModelMeta(nvar, x0=x0, nnzh=nnzh, minimize=true, islp=false, name=name)

  counters = Counters()

  return RADNLPModel(meta, counters, f, x->T[], ∇f!, cfg, cfH, nothing)
end

function RADNLPModel(f        :: Function, 
                     x0       :: AbstractVector{T}, 
                     lvar     :: AbstractVector, 
                     uvar     :: AbstractVector;
                     name     :: String = "Generic",
                     gradient :: Function = smart_reverse,
                     kwargs...) where T
                    
  nvar = length(x0)
  @lencheck nvar x0 lvar uvar

  (∇f!, cfg) = gradient(x0, f)
  nnzh, cfH = _meta_function(f, x0, nvar)

  meta = NLPModelMeta(nvar, x0 = x0, lvar = lvar, uvar = uvar, nnzh = nnzh, 
                      minimize = true, islp = false, name = name)
  
  counters = Counters()
  
  return RADNLPModel(meta, counters, f, x->T[], ∇f!, cfg, cfH, nothing)
end

function RADNLPModel(f        :: Function, 
                     x0       :: AbstractVector{T},
                     c        :: Function,
                     lcon     :: AbstractVector,
                     ucon     :: AbstractVector; 
                     name     :: String="GenericADNLPModel",
                     y0       :: AbstractVector=fill!(similar(lcon), zero(T)),
                     lin      :: AbstractVector{<: Integer}=Int[], 
                     gradient :: Function = smart_reverse, kwargs...) where T

  nvar = length(x0)
  ncon = length(lcon)
  @lencheck nvar x0
  @lencheck ncon ucon y0
  
  (∇f!, cfg) = gradient(x0, f)
  nnzh, cfH = _meta_function(f, x0, nvar)
  nnzj, cfJ = _meta_function(c, x0, ncon, nvar)

  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0=x0, ncon=ncon, y0=y0, lcon=lcon, ucon=ucon,
                      nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true,
                      islp=false, name=name)

  counters = Counters()

  return RADNLPModel(meta, counters, f, c, ∇f!, cfg, cfH, cfJ)
end

function NLPModels.obj(model :: RADNLPModel, x :: AbstractVector)
  @lencheck model.meta.nvar x
  increment!(model, :neval_obj)
  return model.f(x)
end

function NLPModels.grad!(model :: RADNLPModel, x :: AbstractVector, g :: AbstractVector)
  @lencheck model.meta.nvar x g
  increment!(model, :neval_grad)
  model.∇f!(g, x, model.cfg)
  return g
end

function NLPModels.cons!(model :: RADNLPModel, x :: AbstractVector, c :: AbstractVector)
  @lencheck model.meta.nvar x
  @lencheck model.meta.ncon c
  increment!(model, :neval_cons)
  #c .= model.c(x)
  return model.c(c, x)
end

function NLPModels.jac_structure!(model :: RADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck model.meta.nnzj rows cols
  I, J, _ = findnz(model.cfJ.sparsity)
  return I, J
end

function NLPModels.jac_coord!(model :: RADNLPModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck model.meta.nvar x
  @lencheck model.meta.nnzj vals
  increment!(model, :neval_jac)
  J = similar(model.cfJ.sparsity) #How to really use vals?
  forwarddiff_color_jacobian!(J, model.c, x, model.cfJ)
  vals .= J.nzval
  return vals
end

function NLPModels.jac(model :: RADNLPModel, x :: AbstractVector)
  @lencheck model.meta.nvar x
  increment!(model, :neval_jac)
  J = similar(model.cfJ.sparsity)
  forwarddiff_color_jacobian!(J, model.c, x, model.cfJ)
  return J
end

function NLPModels.jprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck model.meta.nvar x v
  @lencheck model.meta.ncon Jv
  increment!(model, :neval_jprod)
  cache1 = ForwardDiff.Dual{SparseDiffTools.DeivVecTag}.(x, v)
  cache2 = ForwardDiff.Dual{SparseDiffTools.DeivVecTag}.(x, v)
  SparseDiffTools.auto_jacvec!(Jv, model.c, x, v, cache1, cache2)
  #Jv .= ForwardDiff.derivative(t -> model.c(x + t * v), 0)
  return Jv
end

function NLPModels.jtprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck model.meta.nvar x Jtv
  @lencheck model.meta.ncon v
  increment!(model, :neval_jtprod)
  dx = similar(model.meta.y0)
  Jtv .= ForwardDiff.gradient(x -> dot(model.c(dx, x), v), x) #ReverseDiff without preparation isn't better
  return Jtv
end

function NLPModels.hess_structure!(model :: RADNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck model.meta.nnzh rows cols
  H = hessian_sparsity(model.f, model.meta.x0)
  rows, cols, _ = findnz(tril(H))
  return rows, cols
end

function NLPModels.hess_coord!(model :: RADNLPModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x
  @lencheck model.meta.nnzh vals
  increment!(model, :neval_hess)
  #J = similar(cfH.sparsity) #Is there a better way here than switching from J to vals??
  #forwarddiff_color_jacobian!(J, model.f, x, model.cfH)
  #vals .= J[:]
  return vals
end

function NLPModels.hess_coord!(model :: RADNLPModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x
  @lencheck model.meta.ncon y
  @lencheck model.meta.nnzh vals
  increment!(model, :neval_hess)
  #...
  return vals
end

function NLPModels.hprod!(model :: RADNLPModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x v Hv
  @lencheck model.meta.ncon y
  increment!(model, :neval_hprod)
  #...
  #=
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  =#
  return Hv
end

function NLPModels.hprod!(model :: RADNLPModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Float64=1.0)
  @lencheck model.meta.nvar x v Hv
  increment!(model, :neval_hprod)
  #...
  #= Option 1
  ℓ(x) = obj_weight * nlp.f(x)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  =#
  #= Option 2
  autoback_hesvec(nlp.f,nlp.meta.x0,nlp.meta.x0)
  =#
  return Hv
end

function hess(nlp :: RADNLPModel, x :: AbstractVector{T}; obj_weight :: Real = one(T)) where T
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_hess)
  if nlp.meta.nnzh == 0
    return spzeros(T,0,0)
  elseif obj_weight == 0.
    rows, cols = hess_structure(nlp)
    return sparse(T, rows, cols, zeros(T, nlp.meta.nnzh), nlp.meta.nvar, nlp.meta.nvar)
  end
  #ℓ(x) = obj_weight * nlp.f(x)
  #Option 1:
  #forwarddiff_color_jacobian(J, x->ForwardDiff.gradient(nlp.f,x), x, nlp.cfH);
  #Option 2:
  Hx = forwarddiff_color_jacobian(J, x->ReverseDiff.gradient(nlp.f,x), x, nlp.cfH);
  #Option 3:
  # Use the gradient config ? and ReverseDiff
  #Option 0: the old one
  #Hx = obj_weight * ForwardDiff.hessian(nlp.f, x)
  return tril(Hx)
end

function hess(nlp :: RADNLPModel, x :: AbstractVector, y :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end