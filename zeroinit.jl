# using LinearAlgebra
using KernelAbstractions
using CUDA
using AMDGPU
using oneAPI
using CUDAKernels
using ROCKernels
using oneAPIKernels
const KA = KernelAbstractions
device = []
T = []
if CUDA.has_cuda_gpu()
    device = CUDADevice()
    T = CuArray
elseif AMDGPU.has_rocm_gpu()
    device = ROCDevice()
    T = ROCArray
else
    device = oneAPIDevice()
    T = oneArray
end

@kernel function localmem_kernel(x)
    I = @index(Global, Linear)
    loc = @localmem Float64 (1000,)
    v = 0
    for i in 1:1000
        v += loc[i]
    end
    x[I] = v
end

x = ones(100) |> T
for i in 1:100
    ev = localmem_kernel(device)(x;ndrange = 100)
    wait(ev)
end
# CUDAKernels: Always 0. @localmem implies a zero initialization

@assert all(x .== 0)

