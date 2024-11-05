from algorithm.functional import vectorize

# The amount of elements to loop through
alias size = 10
# How many Dtype.int32 elements fit into the SIMD register (4 on 128bit, 8 on Cholo's PC)
alias simd_width = simdwidthof[DType.int32]()

fn main():

    var p = DTypePointer[DType.int32]().alloc(size)

    # @parameter allows the closure to capture the `p` pointer
    @parameter
    fn closure[simd_width: Int](i: Int):
        print("storing", simd_width, "els at pos", i)
        p.store[width=simd_width](i, i)

    vectorize[closure, simd_width](size)
    
    print(p.load[width=size]())