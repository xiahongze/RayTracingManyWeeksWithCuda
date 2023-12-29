#pragma once

#include "aabb.h"
#include "hitable.h"
#include <algorithm>
#include <memory>
#include <vector>

int random_int(int min, int max)
{
    // Returns a random integer in [min, max]
    return min + rand() % (max - min + 1);
}

struct bvh_data_node
{
    hitable *obj;
    aabb bbox;
};

bool box_compare(
    const bvh_data_node &a, const bvh_data_node &b, int axis_index)
{
    return a.bbox.axis(axis_index).min < b.bbox.axis(axis_index).min;
}

bool box_x_compare(const bvh_data_node &a, const bvh_data_node &b)
{
    return box_compare(a, b, 0);
}

bool box_y_compare(const bvh_data_node &a, const bvh_data_node &b)
{
    return box_compare(a, b, 1);
}

bool box_z_compare(const bvh_data_node &a, const bvh_data_node &b)
{
    return box_compare(a, b, 2);
}

struct bvh_node : bvh_data_node
{
    int left = -1;
    int right = -1;

    __host__ __device__ bvh_node() {}

    __host__ bvh_node(aabb box)
    {
        bbox = box;
    }

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

    __device__ static void prefill_nodes(bvh_node *nodes, hitable **objects, int list_size)
    {
        for (int i = 0; i < list_size; ++i)
        {
            nodes[i] = bvh_node(objects[i]);
        }
    }

    __device__ static bool hit(const bvh_node *nodes, const ray &r, const interval ray_t, hit_record &rec)
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
                    if (node.obj->hit(r, ray_t, rec))
                    {
                        hit_anything = true;
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

struct _bvh_node
{
    aabb bbox;
    std::shared_ptr<_bvh_node> left;
    std::shared_ptr<_bvh_node> right;
    bvh_data_node data;

    _bvh_node() {}

    _bvh_node(const std::vector<bvh_data_node> &src_objects, size_t start, size_t end)
    {
        auto objects = src_objects; // Create a modifiable array of the source scene objects

        int axis = random_int(0, 2);
        auto comparator = (axis == 0)   ? box_x_compare
                          : (axis == 1) ? box_y_compare
                                        : box_z_compare;

        int object_span = end - start;
        std::cout << "size: " << src_objects.size() << ", start: " << start << ", end: " << end << ", span: " << object_span << std::endl;
        auto a = objects[start];
        std::cout << "a: " << a.bbox.axis(axis).min << ", " << a.bbox.axis(axis).max << std::endl;

        if (object_span == 1)
        {
            // leaf node
            data = objects[start];
            bbox = objects[start].bbox;
            return;
        }

        // if (object_span == 2)
        // {
        //     left = std::make_shared<_bvh_node>();
        //     right = std::make_shared<_bvh_node>();
        //     if (comparator(objects[start], objects[start + 1]))
        //     {
        //         left->data = objects[start];
        //         right->data = objects[start + 1];
        //     }
        //     else
        //     {
        //         left->data = objects[start + 1];
        //         right->data = objects[start];
        //     }
        // }
        // else
        // {
        std::cout << "sorting " << object_span << " objects" << std::endl;
        std::sort(objects.begin() + start, objects.begin() + end, comparator);
        std::cout << "sorted" << std::endl;

        auto mid = start + object_span / 2;
        left = std::make_shared<_bvh_node>(objects, start, mid);
        right = std::make_shared<_bvh_node>(objects, mid, end);
        // }

        bbox = aabb(left->bbox, right->bbox);
    }

    void to_linearized_bvh_node(std::vector<bvh_node> &nodes)
    {
        // add self
        nodes.push_back(bvh_node(bbox));

        // add children
        if (left != nullptr)
        {
            left->to_linearized_bvh_node(nodes);
            nodes.back().left = nodes.size() - 1;
        }
        if (right != nullptr)
        {
            right->to_linearized_bvh_node(nodes);
            nodes.back().right = nodes.size() - 1;
        }
    }
};

__host__ void build_tree(bvh_node *nodes, int size)
{
    // create a vector of bvh_data_nodes
    std::vector<bvh_data_node> data_nodes;
    data_nodes.reserve(size);
    for (int i = 0; i < size; ++i)
    {
        data_nodes.push_back(nodes[i]);
    }
    std::cout << "data nodes created" << std::endl;

    // build the internal tree
    auto root = std::make_shared<_bvh_node>(data_nodes, 0, size);
    std::cout << "bvh tree built" << std::endl;

    // convert to linearized bvh_node
    std::vector<bvh_node> linearized_nodes;
    root->to_linearized_bvh_node(linearized_nodes);
    std::cout << "bvh tree linearized" << std::endl;

    // copy back to nodes
    for (int i = 0; i < linearized_nodes.size(); ++i)
    {
        // print out node info, left right indices
        std::cout << "node " << i << " left: " << linearized_nodes[i].left << ", right: " << linearized_nodes[i].right << std::endl;
        nodes[i] = linearized_nodes[i];
    }
    std::cout << "size of nodes: " << linearized_nodes.size() << std::endl; // should be "size"
    std::cout << "original size: " << size << std::endl;                    // should be "size
    std::cout << "bvh tree copied back" << std::endl;
}