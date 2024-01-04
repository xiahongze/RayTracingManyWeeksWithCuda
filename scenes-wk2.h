#pragma once

#include "bvh.h"
#include "camera.h"
#include "hitable.h"

void earth(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny);