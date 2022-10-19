using PyCall 
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

NestedArray.len(t::TorchTensor)::Int = py"len"(t.v)
Base.size(t::TorchTensor) = py"get_shape"(t.v)
