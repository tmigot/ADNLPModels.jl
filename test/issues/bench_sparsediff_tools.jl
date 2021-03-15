using SparseDiffTools, BenchmarkTools
x = rand(300)
v = rand(300)
f(u) = sum(abs2,u)
du = similar(x)
c1 = similar(x); c2 = similar(x); c3 = similar(x); c4 = similar(x)
cache1 = ForwardDiff.Dual{SparseDiffTools.DeivVecTag}.(x, v)
cache2 = ForwardDiff.Dual{SparseDiffTools.DeivVecTag}.(x, v)
config = ForwardDiff.GradientConfig(f,x)
@btime num_hesvec!($du, $f, $x, $v, $c1, $c3, $c4)
@btime numauto_hesvec!($du, $f, $x, $v, $config, $c1, $c2)
@btime autonum_hesvec!($du, $f, $x, $v, $c1, $cache1, $cache2)
@btime numback_hesvec!(du, f, x, v, $c1, $c2)
@btime autoback_hesvec!(du, f, x, v, $cache1, $cache2)