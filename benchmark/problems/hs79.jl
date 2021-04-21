function hs79_radnlp(; n::Int=100, type::Val{T}=Val(Float64), kwargs...) where T
  return RADNLPModel(
    x->(x[1] - 1)^2 + (x[1] - x[2])^2 + (x[2] - x[3])^2 + (x[3] - x[4])^4 + (x[4] - x[5])^4, 
    2ones(T, 5),
    x->[x[1] + x[2]^2 + x[3]^3 - 2 - 3*sqrt(2); x[2] - x[3]^2 + x[4]   + 2 - 2*sqrt(2); x[1]*x[5] - 2], 
    zeros(3), zeros(3),
    name = "hs79_radnlp"
  )
end

function hs79_autodiff(; n::Int=100, type::Val{T}=Val(Float64), adbackend=ADNLPModels.ForwardDiffAD()) where T
  return ADNLPModel(
    x->(x[1] - 1)^2 + (x[1] - x[2])^2 + (x[2] - x[3])^2 + (x[3] - x[4])^4 + (x[4] - x[5])^4, 
    2ones(T, 5),
    x->[x[1] + x[2]^2 + x[3]^3 - 2 - 3*sqrt(2); x[2] - x[3]^2 + x[4]   + 2 - 2*sqrt(2); x[1]*x[5] - 2], 
    zeros(3), zeros(3),
    name = "hs79_autodiff", adbackend=adbackend
  )
end