#pragma once

#include "bvh.h"
#include "camera.h"
#include "hitable.h"

void earth(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny);

void two_perlin_spheres(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny);

void quads(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny);

void simple_light(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny);

void cornell_box(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny, bool rotate_translate);
