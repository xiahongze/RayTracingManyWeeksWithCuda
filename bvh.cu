#include "bvh.h"
#include <algorithm>
#include <memory>
#include <vector>

int random_int(int min, int max)
{
    // Returns a random integer in [min, max]
    return min + rand() % (max - min + 1);
}

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

struct _bvh_node
{
    aabb bbox;
    std::shared_ptr<_bvh_node> left;
    std::shared_ptr<_bvh_node> right;
    bvh_data_node data;

    _bvh_node() {}

    _bvh_node(std::vector<bvh_data_node> &objects, size_t start, size_t end)
    {
        int axis = random_int(0, 2);
        auto comparator = (axis == 0)   ? box_x_compare
                          : (axis == 1) ? box_y_compare
                                        : box_z_compare;

        int object_span = end - start;

        if (object_span == 1)
        {
            // leaf node
            data = objects[start];
            bbox = objects[start].bbox;
            return;
        }

        std::sort(objects.begin() + start, objects.begin() + end, comparator);
        auto mid = start + object_span / 2;
        left = std::make_shared<_bvh_node>(objects, start, mid);
        right = std::make_shared<_bvh_node>(objects, mid, end);

        bbox = aabb(left->bbox, right->bbox);
    }

    static std::vector<bvh_node> to_linearized_bvh_node(std::shared_ptr<_bvh_node> root, int &tree_height)
    {
        tree_height = 0; // Initialize tree height

        if (root == nullptr)
        {
            return std::vector<bvh_node>();
        }

        std::vector<bvh_node> nodes;
        int index = 0;
        std::vector<std::tuple<std::shared_ptr<_bvh_node>, int, int>> stack; // Add an integer for depth
        stack.push_back(std::make_tuple(root, -1, 0));                       // Start with depth 0

        while (stack.size() > 0)
        {
            auto node = std::get<0>(stack.back());
            auto parent_index = std::get<1>(stack.back());
            int depth = std::get<2>(stack.back()); // Current depth
            stack.pop_back();

            // Update tree height if a greater depth is found
            if (depth > tree_height)
            {
                tree_height = depth;
            }

            nodes.push_back(bvh_node(node->data.obj, node->bbox));
            if (parent_index != -1)
            {
                if (nodes[parent_index].left == -1)
                {
                    nodes[parent_index].left = index;
                }
                else
                {
                    nodes[parent_index].right = index;
                }
            }

            if (node->left != nullptr)
            {
                stack.push_back(std::make_tuple(node->left, index, depth + 1));
            }
            if (node->right != nullptr)
            {
                stack.push_back(std::make_tuple(node->right, index, depth + 1));
            }

            index++;
        }

        return nodes;
    }
};

__host__ int bvh_node::build_tree(bvh_node *nodes, int size)
{
    // create a vector of bvh_data_nodes
    std::vector<bvh_data_node> data_nodes;
    data_nodes.reserve(size);
    for (int i = 0; i < size; ++i)
    {
        data_nodes.push_back(nodes[i]);
    }
    // build the internal tree
    auto root = std::make_shared<_bvh_node>(data_nodes, 0, size);
    std::clog << "bvh tree built" << std::endl;

    // convert to linearized bvh_node
    int tree_height = 0;
    std::vector<bvh_node> linearized_nodes = _bvh_node::to_linearized_bvh_node(root, tree_height);
    std::clog << "bvh tree linearized" << std::endl;

    std::clog << "size of nodes: " << linearized_nodes.size() << std::endl; // should be "size"
    std::clog << "original size: " << size << std::endl;                    // should be "size
    std::clog << "tree height: " << tree_height << std::endl;

    // copy back to nodes
    for (int i = 0; i < linearized_nodes.size(); ++i)
    {
        // print out node info, left right indices
        // std::cout << "node " << i << " left: " << linearized_nodes[i].left << ", right: " << linearized_nodes[i].right << std::endl;
        nodes[i] = linearized_nodes[i];
    }
    std::clog << "bvh tree copied back" << std::endl;

    return tree_height;
}

__device__ bool bvh_node::hit(const bvh_node *nodes, const ray &r, interval ray_t, hit_record &rec)
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
                    ray_t = interval(ray_t.min, rec.t);
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

__device__ void bvh_node::prefill_nodes(bvh_node *nodes, hitable **objects, int list_size)
{
    for (int i = 0; i < list_size; ++i)
    {
        nodes[i] = bvh_node(objects[i]);
    }
}