#pragma once

#include "bvh.h"
#include "camera.h"
#include "hitable.h"

void random_spheres(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable *&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny, bool bounce, float bounce_pct, bool checkered);
