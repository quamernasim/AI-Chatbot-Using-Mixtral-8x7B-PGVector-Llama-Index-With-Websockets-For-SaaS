import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

# CUDA kernel
mod = SourceModule("""
__global__ void add(int *a, int *b, int *c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}
""")

add_gpu = mod.get_function("add")

# Host arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.zeros_like(a)

# Allocate device memory
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Transfer data to device
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Execute kernel
add_gpu(a_gpu, b_gpu, c_gpu, block=(3, 1, 1))

# Transfer results to host
cuda.memcpy_dtoh(c, c_gpu)
print(c)
