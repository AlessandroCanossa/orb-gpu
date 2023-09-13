#include "orb_gpu/cuda.cuh"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

struct if_corner
{
    __host__ __device__ bool operator()(Corner x) { return x.isCorner; }
};

void collect(Corner* kpts, int size, Corner* out, cudaStream_t aStream)
{
    thrust::device_ptr<Corner> prova = thrust::device_pointer_cast(kpts);
    thrust::device_ptr<Corner> output = thrust::device_pointer_cast(out);

    thrust::copy_if(thrust::cuda::par.on(aStream), prova, prova + size, output, if_corner());
}

__global__ void toKeypoints(Corner* aCorners, int aSize, Keypoint* aKeypoints)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id > aSize) {
        return;
    }

    aKeypoints[id].corner = aCorners[id];
}

void toKpts(Corner* aCorners, int aSize, Keypoint* aKeypoints, cudaStream_t aStream)
{
    toKeypoints<<<ceil((double)aSize / 32), 32, 0, aStream>>>(aCorners, aSize, aKeypoints);
}
