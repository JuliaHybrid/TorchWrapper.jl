using PyCall 
using NestedArray
import NestedArray
import Base
py"""
import torch

def get_shape(t):
    return t.shape
"""


struct TorchTensor 
    v::PyObject
end

struct PyFunction 
    v::PyObject
end

(>>)(t::TorchTensor, f::PyFunction) = f.v(t.v)


NestedArray.len(t::TorchTensor)::Int = t >> PyFunction(py"len")
Base.size(t::TorchTensor) = t >> PyFunction(py"get_shape")
Base.rand(::Type{TorchTensor}, shape::Tuple{Vararg{Int64, N}} where N) = TorchTensor(py"torch.rand"(shape...))


