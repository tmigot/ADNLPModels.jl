function dixon3dq_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  function f(x)
    n = length(x)
    return (x[1] - 1.0)^2 + (x[n] - 1.0)^2 + sum((x[i] - x[i+1])^2 for i=2:n-1)
  end
  x0 = -ones(T, n)
  return RADNLPModel(f, x0, name="dixon3dq_radnlp"; kwargs...)
end

function dixon3dq_autodiff(; n::Int=100, type::Val{T}=Val(Float64)) where T
  function f(x)
    n = length(x)
    return (x[1] - 1.0)^2 + (x[n] - 1.0)^2 + sum((x[i] - x[i+1])^2 for i=2:n-1)
  end
  x0 = -ones(T, n)
  return ADNLPModel(f, x0, name="dixon3dq_autodiff")
end