#pragma once

#include "bvh.h"
#include "camera.h"
#include "hitable.h"

void final_scene_wk3(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, hitable_list *&d_lights, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny, int rand_seed);
