#pragma once

#include "aabb.h"
#include "hitable.h"
#include <vector>

struct bvh_node
{
    aabb bbox;
    int left = -1;
    int right = -1;
    hitable *data;

    __host__ __device__ bvh_node() {}

    __device__ bvh_node(hitable *object)
    {
        data = object;
        bbox = object->bounding_box();
    }

    __device__ static void prefill_nodes(bvh_node *nodes, hitable **objects, int list_size)
    {
        for (int i = 0; i < list_size; ++i)
        {
            nodes[i] = bvh_node(objects[i]);
        }
    }

    __host__ static int build(std::vector<bvh_node> &nodes, int start, int end)
    {
        // Base case: single node
        if (end - start == 1)
        {
            return start;
        }

        // Compute the bounding box of all nodes in range
        aabb box;
        for (int i = start; i < end; ++i)
        {
            box = aabb(box, nodes[i].bbox);
        }

        // Determine the longest axis (0: x, 1: y, 2: z)
        int axis = box.longest_axis();

        // Split nodes around the median on the chosen axis
        int mid = (start + end) / 2;
        std::nth_element(nodes.begin() + start, nodes.begin() + mid, nodes.begin() + end,
                         [axis](const bvh_node &a, const bvh_node &b)
                         {
                             return a.bbox.center()[axis] < b.bbox.center()[axis];
                         });

        // Recursively build left and right subtrees
        int left = build(nodes, start, mid);
        int right = build(nodes, mid, end);

        // Create new node encompassing both subtrees
        bvh_node node;
        node.bbox = aabb(nodes[left].bbox, nodes[right].bbox);
        node.left = left;
        node.right = right;
        nodes.push_back(node);

        return nodes.size() - 1; // Return index of the new node
    }

    __device__ static bool hit(const bvh_node *nodes, const ray &r, const interval ray_t, hit_record &rec) const
    {
        // Stack for node indices
        int stack[64]; // Adjust size as needed
        int stackPtr = 0;

        // Push index of root node
        stack[stackPtr++] = 0;

        bool hit_anything = false;
        while (stackPtr > 0)
        {
            // Pop a node from the stack
            int nodeIdx = stack[--stackPtr];
            const bvh_node &node = nodes[nodeIdx];

            if (node.bbox.hit(r, ray_t))
            {
                // If it's a leaf node
                if (node.left == -1 && node.right == -1)
                {
                    hit_record temp_rec;
                    if (node.data->hit(r, interval(0.001, ray_t.max), temp_rec))
                    {
                        hit_anything = true;
                        ray_t.max = temp_rec.t; // Update the closest hit so far
                        rec = temp_rec;
                    }
                }
                else
                {
                    // Not a leaf node, push children to stack
                    if (node.left != -1)
                        stack[stackPtr++] = node.left;
                    if (node.right != -1)
                        stack[stackPtr++] = node.right;
                }
            }
        }

        return hit_anything;
    }
};
