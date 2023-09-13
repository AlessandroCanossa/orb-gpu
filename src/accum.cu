#include "orb_gpu/cuda.cuh"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

struct if_even
{
    __host__ __device__ bool operator()(Kp x) { return x.isCorner; }
};

void collect(Kp* kpts, int size, Kp* out, cudaStream_t aStream)
{
    thrust::device_ptr<Kp> prova = thrust::device_pointer_cast(kpts);
    thrust::device_ptr<Kp> output = thrust::device_pointer_cast(out);

    thrust::copy_if(thrust::cuda::par.on(aStream), prova, prova + size, output, if_even());
}
