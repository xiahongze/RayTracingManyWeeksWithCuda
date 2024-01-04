#include "material.h"
#include "scenes-wk2.h"
#include "sphere.h"
#include "texture.h"
#include "utils.h"

__global__ void create_earth(bvh_node *d_bvh_nodes, hitable **d_list, camera *d_camera,
                             unsigned char *d_pixel_data, int width, int height, int channels,
                             int list_size, int nx, int ny)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i > 0) || (j > 0))
        return;

    auto *earth_texture = new rtapp::image_texture(d_pixel_data, width, height, channels);
    d_list[0] = new sphere(vec3(0, 0, 0), 2.0, new lambertian(earth_texture));

    // create bvh_nodes
    bvh_node::prefill_nodes(d_bvh_nodes, d_list, list_size);

    *d_camera = camera();
    d_camera->lookfrom = vec3(-9, -2, -10);
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
    auto earth_texture = rtapp::image_texture("assets/earthmap.jpg");

    // copy texture to device
    unsigned char *d_pixel_data;
    checkCudaErrors(cudaMalloc((void **)&d_pixel_data, earth_texture.pixel_data_size));
    checkCudaErrors(cudaMemcpy(d_pixel_data, earth_texture.pixel_data, earth_texture.pixel_data_size, cudaMemcpyHostToDevice));

    list_size = 1;
    checkCudaErrors(cudaMalloc((void **)&d_list, list_size * sizeof(hitable *)));

    tree_size = 2 * list_size;
    h_bvh_nodes = new bvh_node[tree_size]; // binary tree
    checkCudaErrors(cudaMalloc((void **)&d_bvh_nodes, tree_size * sizeof(bvh_node)));

    create_earth<<<dim3(1, 1), dim3(1, 1)>>>(d_bvh_nodes, d_list, d_camera,
                                             d_pixel_data, earth_texture.width, earth_texture.height, earth_texture.channels,
                                             list_size, nx, ny);

    std::cout << "earth scene created" << std::endl;
}