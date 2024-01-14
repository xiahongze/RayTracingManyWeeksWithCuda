#pragma once

#include "hitable.h"

#define RND (curand_uniform(&local_rand_state))

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

#define INIT_LIST_AND_TREE(size)                                                  \
    list_size = (size);                                                           \
    checkCudaErrors(cudaMalloc((void **)&d_list, list_size * sizeof(hitable *))); \
    tree_size = 2 * list_size;                                                    \
    h_bvh_nodes = new bvh_node[tree_size]; /* binary tree */                      \
    checkCudaErrors(cudaMalloc((void **)&d_bvh_nodes, tree_size * sizeof(bvh_node)));

#define CHECK_SINGLE_THREAD_BOUNDS()               \
    int i = threadIdx.x + blockIdx.x * blockDim.x; \
    int j = threadIdx.y + blockIdx.y * blockDim.y; \
    if ((i > 0) || (j > 0))                        \
        return;

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

__global__ void free_objects(hitable **d_list, int size);

__device__ float degrees_to_radians(float degrees);

#define INIT_RAND_LOCAL()         \
    curandState local_rand_state; \
    curand_init(rand_seed, 0, 0, &local_rand_state);
