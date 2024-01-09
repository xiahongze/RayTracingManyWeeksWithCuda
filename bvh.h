#pragma once

#include "aabb.h"
#include "hitable.h"

struct bvh_data_node
{
    hitable *obj;
    aabb bbox;
};

struct bvh_node : bvh_data_node
{
    int left = -1;
    int right = -1;

    __host__ __device__ bvh_node() {}

    __host__ bvh_node(hitable *object, aabb box)
    {
        obj = object;
        bbox = box;
    }

    __device__ bvh_node(hitable *object)
    {
        obj = object;
        bbox = object->bounding_box();
    }

    __device__ static void prefill_nodes(bvh_node *nodes, hitable *objects, int list_size);

    __device__ static bool hit(const bvh_node *nodes, const ray &r, interval ray_t, hit_record &rec);

    /**
     * @brief _bvh_node is a helper class for building the bvh tree
     *
     * @param nodes data nodes copy from the device and then be updated with the tree structure. After this function, nodes will be the linearized bvh tree
     * @param size size of the nodes
     *
     * @return the height of the tree
     */
    __host__ static int build_tree(bvh_node *nodes, int size);
};
