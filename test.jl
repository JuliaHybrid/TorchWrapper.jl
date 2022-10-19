include("src/torchWrapper.jl")

a = rand(TorchTensor, (2,3))
@show a 
@show len(a)
@show size(a)