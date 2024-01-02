#pragma once

#include "hitable.h"

#ifndef RAND_SEED
#define RAND_SEED 1984
#endif

#define RND (curand_uniform(&local_rand_state))

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

__global__ void free_objects(hitable **d_list, int size);
