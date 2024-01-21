#pragma once

#include "bvh.h"
#include "camera.h"

__global__ void render(vec3 *d_fb, int max_x, int max_y, int ns, int max_depth, int rand_seed, camera *d_camera, bvh_node *d_bvh_nodes, hitable_list **d_lights);