function curly10_radnlp(; n::Int=100,  type::Val{T}=Val(Float64), kwargs...) where T
  n < 2 && @warn("curly: number of variables must be ≥ 2")
  n = max(2, n)
  b = 10
  function f(x)
    n = length(x)
    return sum((sum(x[j] for j=i:min(i+b,n))) * ((sum(x[j] for j=i:min(i+b,n))) * ((sum(x[j] for j=i:min(i+b,n)))^2 - 20) - 0.1) for i = 1:n)
  end
  x0 = T[1.0e-4 * i /(n+1) for i = 1:n]
  return RADNLPModel(f, x0, name="curly10_radnlp"; kwargs...)
end

function curly10_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  n < 2 && @warn("curly: number of variables must be ≥ 2")
  n = max(2, n)
  b = 10
  function f(x)
    n = length(x)
    return sum((sum(x[j] for j=i:min(i+b,n))) * ((sum(x[j] for j=i:min(i+b,n))) * ((sum(x[j] for j=i:min(i+b,n)))^2 - 20) - 0.1) for i = 1:n)
  end
  x0 = T[1.0e-4 * i /(n+1) for i = 1:n]
  return ADNLPModel(f, x0, name="curly10_autodiff")
end
