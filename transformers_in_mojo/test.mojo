from max.tensor import Tensor, TensorShape
from algorithm import cumsum
from utils.index import Index
from time import now
from random import seed

fn ones(d1: Int, d2: Int) -> Tensor[DType.float32]:
    var shape = TensorShape(d1, d2)
    var out = Tensor[DType.float32].rand(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out[Index(i, j)] = 1
    return out

fn tril(mat: Tensor[DType.float32]) -> Tensor[DType.float32]:
    var shape = mat.shape()
    var out = Tensor[DType.float32].rand(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            var val = mat[Index(i, j)]
            if j > i:
                val = 0
            out[Index(i, j)] = val
    return out

fn main():
    var start_time = now()

    seed(42)
    var a: Tensor[DType.float32] = tril(ones(3,3))
    # a = a / sum(a, 1, keepdim=True)




    var elapsed_time_ms = (now() - start_time) / 1_000_000.0
    print(elapsed_time_ms)

