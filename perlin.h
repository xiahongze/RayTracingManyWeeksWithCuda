#pragma once
#include "vec3.h"
#include <curand.h>
#include <curand_kernel.h>

class perlin
{
public:
    __device__ perlin(curandState *local_rand_state);

    __device__ ~perlin();

    __device__ float noise(const vec3 &p) const;

    __device__ float turb(const vec3 &p, int depth = 7) const;

private:
    static const int point_count = 256;
    vec3 *ranvec;
    int *perm_x;
    int *perm_y;
    int *perm_z;

    __device__ static int *perlin_generate_perm(curandState *local_rand_state);

    __device__ static void permute(int *p, int n, curandState *local_rand_state);

    __device__ static float perlin_interp(vec3 c[2][2][2], float u, float v, float w);
};
