using PyCall 
using NestedArray
import NestedArray
import EasyMonad
import Base
py"""
import torch

def get_shape(t):
    return t.shape
"""

abstract type TorchType end

abstract type TorchInput <: TorchType end

struct TorchTensor <:TorchInput
    v::PyObject
end

struct PyFunction <:TorchType
    v::PyObject
end

struct TorchDevice <: TorchInput 
    v::Union{String, Int}
end

ignorable_fetch(t) = begin 
    !(t isa TorchType) && return t 
    return t.v
end

import EasyMonad.(>>)
(>>)(t::TorchInput, f::PyFunction) = f.v(t.v)
(>>)(t::Tuple, f::PyFunction) = begin 
    nt = map(ignorable_fetch, t)
    return f.v(nt...)
end

NestedArray.len(t::TorchTensor)::Int = t >> PyFunction(py"len")
Base.size(t::TorchTensor) = t >> PyFunction(py"get_shape")
Base.rand(::Type{TorchTensor}, shape::Tuple{Vararg{Int64, N}} where N) = TorchTensor(py"torch.rand"(shape...))


