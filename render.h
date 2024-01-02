#pragma once

#include "bvh.h"
#include "camera.h"
#include "curand_kernel.h"

#ifndef RAY_MAX_DEPTH
#define RAY_MAX_DEPTH 50
#endif

__device__ vec3 get_ray_color_pixel(const ray &r, bvh_node *d_bvh_nodes, curandState *local_rand_state);

__global__ void render(vec3 *d_fb, int max_x, int max_y, int ns, camera *d_camera, bvh_node *d_bvh_nodes);