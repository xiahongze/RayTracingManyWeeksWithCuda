#include "material.h"
#include "scenes-wk2.h"
#include "sphere.h"
#include "texture.h"
#include "utils.h"

__global__ void create_earth(bvh_node *d_bvh_nodes, hitable **d_list, camera *d_camera, rtapp::image_texture *d_earth_texture, int list_size, int nx, int ny)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i > 0) || (j > 0))
        return;

    d_list[0] = new sphere(vec3(0, 0, 0), 2.0, new lambertian(d_earth_texture));

    // create bvh_nodes
    bvh_node::prefill_nodes(d_bvh_nodes, d_list, list_size);

    *d_camera = camera();
    d_camera->lookfrom = vec3(0, 0, 12);
    d_camera->lookat = vec3(0, 0, 0);
    d_camera->vup = vec3(0, 1, 0);
    d_camera->vfov = 20.0;
    d_camera->image_width = nx;
    d_camera->image_height = ny;
    d_camera->defocus_angle = 0.0;
    d_camera->initialize();
}

void earth(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny)
{
    std::cout << "creating earth scene..." << std::endl;
    auto earth_texture = rtapp::image_texture("assets/earthmap.jpg");
    std::cout << "earth texture loaded" << std::endl;
    // auto v = earth_texture.value(0, 0, vec3(0, 0, 0));
    // std::cout << "earth texture value: " << v << std::endl;

    auto d_earth_texture = earth_texture.to_device();
    std::cout << "earth texture copied to device" << std::endl;

    list_size = 1;
    checkCudaErrors(cudaMalloc((void **)&d_list, list_size * sizeof(hitable *)));

    tree_size = 2 * list_size;
    h_bvh_nodes = new bvh_node[tree_size]; // binary tree
    checkCudaErrors(cudaMalloc((void **)&d_bvh_nodes, tree_size * sizeof(bvh_node)));

    create_earth<<<dim3(1, 1), dim3(1, 1)>>>(d_bvh_nodes, d_list, d_camera, d_earth_texture, list_size, nx, ny);

    std::cout << "earth scene created" << std::endl;
}