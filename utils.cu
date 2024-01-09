#include "utils.h"
#include <iostream>

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void free_objects(hitable **d_list, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0)
    {
        for (int i = 0; i < size; i++)
        {
            delete d_list[i];
        }
    }
}

__device__ float degrees_to_radians(float degrees)
{
    return degrees * ((float)M_PI) / 180.0f;
}
