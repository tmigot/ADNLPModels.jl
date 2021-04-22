export SimADNLPModel

function simplified_function(fc :: Function)
  @variables xs[1:n]
  _fun = Symbolics.simplify(fc(xs))
  _fun = eval(Symbolics.build_function(_fun, xs, expression = Val{false}))
  return x -> Base.invokelatest(_fun, x)
end

"""
    SimADNLPModel(f, x0)
    SimADNLPModel(f, x0, lvar, uvar)
    SimADNLPModel(f, x0, c, lcon, ucon)
    SimADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)

SimADNLPModel is an AbstractNLPModel using automatic differentiation to compute the derivatives.
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
function SimADNLPModel(f, x0::AbstractVector{T}; name::String = "Generic", adbackend = ForwardDiffAD()) where T
  nvar = length(x0)
  @lencheck nvar x0

  nnzh = nvar * (nvar + 1) / 2

  meta = NLPModelMeta(nvar, x0=x0, nnzh=nnzh, minimize=true, islp=false, name=name)

  return ADNLPModel(meta, Counters(), adbackend, simplified_function(f), x->T[])
end

function SimADNLPModel(f, x0::AbstractVector{T}, lvar::AbstractVector, uvar::AbstractVector;
                    name::String = "Generic", adbackend = ForwardDiffAD()) where T
  nvar = length(x0)
  @lencheck nvar x0 lvar uvar

  nnzh = nvar * (nvar + 1) / 2

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, nnzh=nnzh, minimize=true, islp=false, name=name)

  return ADNLPModel(meta, Counters(), adbackend, simplified_function(f), x->T[])
end

function SimADNLPModel(f, x0::AbstractVector{T}, c, lcon::AbstractVector, ucon::AbstractVector;
                    y0::AbstractVector=fill!(similar(lcon), zero(T)),
                    name::String = "Generic", lin::AbstractVector{<: Integer}=Int[], adbackend = ForwardDiffAD()) where T

  nvar = length(x0)
  ncon = length(lcon)
  @lencheck nvar x0
  @lencheck ncon ucon y0

  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon

  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0=x0, ncon=ncon, y0=y0, lcon=lcon, ucon=ucon,
    nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true, islp=false, name=name)

  return ADNLPModel(meta, Counters(), adbackend, simplified_function(f), simplified_function(c))
end

function SimADNLPModel(f, x0::AbstractVector{T}, lvar::AbstractVector, uvar::AbstractVector,
                    c, lcon::AbstractVector, ucon::AbstractVector;
                    y0::AbstractVector=fill!(similar(lcon), zero(T)),
                    name::String = "Generic", lin::AbstractVector{<: Integer}=Int[], adbackend = ForwardDiffAD()) where T

  nvar = length(x0)
  ncon = length(lcon)
  @lencheck nvar x0 lvar uvar
  @lencheck ncon y0 ucon

  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon

  nln = setdiff(1:ncon, lin)

  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0,
    lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true,
    islp=false, name=name)

  return ADNLPModel(meta, Counters(), adbackend, simplified_function(f), simplified_function(c))
end
