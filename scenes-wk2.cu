#include "constant_medium.h"
#include "material.h"
#include "quad.h"
#include "scenes-wk2.h"
#include "sphere.h"
#include "texture.h"
#include "utils.h"

__global__ void create_earth(bvh_node *d_bvh_nodes, hitable **d_list, camera *d_camera,
                             unsigned char *d_pixel_data, int width, int height, int channels,
                             int list_size, int nx, int ny)
{
    CHECK_SINGLE_THREAD_BOUNDS();

    auto earth_texture = new rtapp::image_texture(d_pixel_data, width, height, channels);
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

#define LOAD_IMAGE_TEXTURE(path)                                                  \
    auto texture = rtapp::image_texture(path);                                    \
    unsigned char *d_pixel_data;                                                  \
    checkCudaErrors(cudaMalloc((void **)&d_pixel_data, texture.pixel_data_size)); \
    checkCudaErrors(cudaMemcpy(d_pixel_data, texture.pixel_data, texture.pixel_data_size, cudaMemcpyHostToDevice));

void earth(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny)
{
    LOAD_IMAGE_TEXTURE("assets/earthmap.jpg");

    list_size = 1;
    checkCudaErrors(cudaMalloc((void **)&d_list, list_size * sizeof(hitable *)));

    tree_size = 2 * list_size;
    h_bvh_nodes = new bvh_node[tree_size]; // binary tree
    checkCudaErrors(cudaMalloc((void **)&d_bvh_nodes, tree_size * sizeof(bvh_node)));

    create_earth<<<dim3(1, 1), dim3(1, 1)>>>(d_bvh_nodes, d_list, d_camera,
                                             d_pixel_data, texture.width, texture.height, texture.channels,
                                             list_size, nx, ny);

    std::cout << "earth scene created" << std::endl;
}

__global__ void create_two_perlin_spheres(bvh_node *d_bvh_nodes, hitable **d_list, camera *d_camera,
                                          int list_size, int nx, int ny)
{
    CHECK_SINGLE_THREAD_BOUNDS();
    INIT_RAND_LOCAL();

    auto perlin_texture = new rtapp::noise_texture(4.0, &local_rand_state);
    d_list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(perlin_texture));
    d_list[1] = new sphere(vec3(0, 2, 0), 2, new lambertian(perlin_texture));

    // create bvh_nodes
    bvh_node::prefill_nodes(d_bvh_nodes, d_list, list_size);

    *d_camera = camera();
    d_camera->lookfrom = vec3(13, 2, 3);
    d_camera->lookat = vec3(0, 0, 0);
    d_camera->vup = vec3(0, 1, 0);
    d_camera->vfov = 20.0;
    d_camera->image_width = nx;
    d_camera->image_height = ny;
    d_camera->defocus_angle = 0.0;
    d_camera->initialize();
}

void two_perlin_spheres(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny)
{
    INIT_LIST_AND_TREE(2);

    create_two_perlin_spheres<<<dim3(1, 1), dim3(1, 1)>>>(d_bvh_nodes, d_list, d_camera,
                                                          list_size, nx, ny);
}

__global__ void create_quads(bvh_node *d_bvh_nodes, hitable **d_list, camera *d_camera,
                             int list_size, int nx, int ny)
{
    CHECK_SINGLE_THREAD_BOUNDS();

    // Materials
    auto left_red = new lambertian(vec3(1.0, 0.2, 0.2));
    auto back_green = new lambertian(vec3(0.2, 1.0, 0.2));
    auto right_blue = new lambertian(vec3(0.2, 0.2, 1.0));
    auto upper_orange = new lambertian(vec3(1.0, 0.5, 0.0));
    auto lower_teal = new lambertian(vec3(0.2, 0.8, 0.8));

    // Quads
    d_list[0] = new quad(vec3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0), left_red);
    d_list[1] = new quad(vec3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0), back_green);
    d_list[2] = new quad(vec3(3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0), right_blue);
    d_list[3] = new quad(vec3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), upper_orange);
    d_list[4] = new quad(vec3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4), lower_teal);

    // create bvh_nodes
    bvh_node::prefill_nodes(d_bvh_nodes, d_list, list_size);

    *d_camera = camera();
    d_camera->lookfrom = vec3(0, 0, 9);
    d_camera->lookat = vec3(0, 0, 0);
    d_camera->vup = vec3(0, 1, 0);
    d_camera->vfov = 80.0;
    d_camera->image_width = nx;
    d_camera->image_height = ny;
    d_camera->defocus_angle = 0.0;
    d_camera->initialize();
}

void quads(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny)
{
    INIT_LIST_AND_TREE(5);

    create_quads<<<dim3(1, 1), dim3(1, 1)>>>(d_bvh_nodes, d_list, d_camera,
                                             list_size, nx, ny);
}

__global__ void create_simple_light(bvh_node *d_bvh_nodes, hitable **d_list, camera *d_camera,
                                    int list_size, int nx, int ny)
{
    CHECK_SINGLE_THREAD_BOUNDS();
    INIT_RAND_LOCAL();

    auto pertext = new rtapp::noise_texture(4, &local_rand_state);
    d_list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(pertext));
    d_list[1] = new sphere(vec3(0, 2, 0), 2, new lambertian(pertext));

    auto difflight = new diffuse_light(vec3(4, 4, 4));
    d_list[2] = new sphere(vec3(0, 7, 0), 2, difflight);
    d_list[3] = new quad(vec3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), difflight);

    // create bvh_nodes
    bvh_node::prefill_nodes(d_bvh_nodes, d_list, list_size);

    *d_camera = camera();
    d_camera->lookfrom = vec3(26, 3, 6);
    d_camera->lookat = vec3(0, 2, 0);
    d_camera->vup = vec3(0, 1, 0);
    d_camera->vfov = 20.0;
    d_camera->image_width = nx;
    d_camera->image_height = ny;
    d_camera->defocus_angle = 0.0;
    d_camera->background = vec3(0, 0, 0);
    d_camera->initialize();
}

void simple_light(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny)
{
    INIT_LIST_AND_TREE(4);

    create_simple_light<<<dim3(1, 1), dim3(1, 1)>>>(d_bvh_nodes, d_list, d_camera,
                                                    list_size, nx, ny);
}

__global__ void create_cornell_box(bvh_node *d_bvh_nodes, hitable **d_list, camera *d_camera,
                                   int list_size, int nx, int ny, bool rotate_translate, bool smoke)
{
    CHECK_SINGLE_THREAD_BOUNDS();

    INIT_RAND_LOCAL();

    auto red = new lambertian(vec3(0.65, 0.05, 0.05));
    auto white = new lambertian(vec3(0.73, 0.73, 0.73));
    auto green = new lambertian(vec3(0.12, 0.45, 0.15));
    auto light = new diffuse_light(vec3(15, 15, 15));

    d_list[0] = new quad(vec3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green);
    d_list[1] = new quad(vec3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red);
    d_list[2] = new quad(vec3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), light);
    d_list[3] = new quad(vec3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white);
    d_list[4] = new quad(vec3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), white);
    d_list[5] = new quad(vec3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0.1), white);

    if (rotate_translate)
    {
        auto box1 = new box(vec3(0, 0, 0), vec3(165, 330, 165), white, &local_rand_state);
        auto box2 = new box(vec3(0, 0, 0), vec3(165, 165, 165), white, &local_rand_state);
        d_list[6] = new translate(new rotate_y(box1, 15), vec3(265, 0, 295));
        d_list[7] = new translate(new rotate_y(box2, -18), vec3(130, 0, 65));
    }
    else
    {
        d_list[6] = new box(vec3(130, 0, 65), vec3(295, 165, 230), white, &local_rand_state);
        d_list[7] = new box(vec3(265, 0, 295), vec3(431, 331, 461), white, &local_rand_state);
    }

    if (smoke)
    {
        d_list[6] = new constant_medium(d_list[6], 0.01, vec3(0, 0, 0));
        d_list[7] = new constant_medium(d_list[7], 0.01, vec3(1, 1, 1));
    }

    // create bvh_nodes
    bvh_node::prefill_nodes(d_bvh_nodes, d_list, list_size);

    *d_camera = camera();
    d_camera->lookfrom = vec3(278, 278, -800);
    d_camera->lookat = vec3(278, 278, 0);
    d_camera->vup = vec3(0, 1, 0);
    d_camera->vfov = 40.0;
    d_camera->image_width = nx;
    d_camera->image_height = ny;
    d_camera->defocus_angle = 0.0;
    d_camera->background = vec3(0, 0, 0);
    d_camera->initialize();
}

void cornell_box(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny, bool rotate_translate, bool smoke)
{
    INIT_LIST_AND_TREE(8);

    create_cornell_box<<<dim3(1, 1), dim3(1, 1)>>>(d_bvh_nodes, d_list, d_camera,
                                                   list_size, nx, ny, rotate_translate, smoke);
}

__global__ void create_final_scene_wk2(bvh_node *d_bvh_nodes, hitable **d_list, camera *d_camera, unsigned char *d_pixel_data,
                                       int width, int height, int channels, int list_size, int nx, int ny)
{
    CHECK_SINGLE_THREAD_BOUNDS();
    int z = 0;

    auto ground = new lambertian(vec3(0.48, 0.83, 0.53));
    int boxes_per_side = 20;
    INIT_RAND_LOCAL()

    for (i = 0; i < boxes_per_side; i++)
    {
        for (j = 0; j < boxes_per_side; j++)
        {
            auto w = 100.0;
            auto x0 = -1000.0 + i * w;
            auto y0 = 0.0;
            auto z0 = -1000.0 + j * w;
            auto x1 = x0 + w;
            auto y1 = 100.0 * curand_uniform(&local_rand_state) - 49;
            auto z1 = z0 + w;
            d_list[z++] = new box(vec3(x0, y0, z0), vec3(x1, y1, z1), ground, &local_rand_state);
        }
    }

    auto light = new diffuse_light(vec3(7, 7, 7));
    auto quad_light = new quad(vec3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light);
    d_list[z++] = quad_light;

    auto center1 = vec3(400, 400, 200);
    auto center2 = center1 + vec3(30, 0, 0);
    auto sphere_material = new lambertian(vec3(0.7, 0.3, 0.1));
    auto moving_sphere = new sphere(center1, center2, 50, sphere_material);
    d_list[z++] = moving_sphere;

    auto sphere2 = new sphere(vec3(260, 150, 45), 50, new dielectric(1.5));
    auto sphere3 = new sphere(vec3(0, 150, 145), 50, new metal(vec3(0.8, 0.8, 0.9), 1.0));
    d_list[z++] = sphere2;
    d_list[z++] = sphere3;

    auto sphere4 = new sphere(vec3(360, 150, 145), 70, new dielectric(1.5));
    auto sphere5 = new sphere(vec3(0, 0, 0), 5000, new dielectric(1.5));
    auto smoke1 = new constant_medium(sphere4, 0.2, vec3(0.2, 0.4, 0.9));
    auto smoke2 = new constant_medium(sphere5, 0.0001, vec3(1, 1, 1));
    // we need to add both the original spheres and the smoke
    d_list[z++] = sphere4;
    d_list[z++] = sphere5;
    d_list[z++] = smoke1;
    d_list[z++] = smoke2;

    auto earth_texture = new rtapp::image_texture(d_pixel_data, width, height, channels);
    auto earth = new sphere(vec3(400, 200, 400), 100, new lambertian(earth_texture));
    d_list[z++] = earth;

    auto pertext = new rtapp::noise_texture(0.1, &local_rand_state);
    auto noise_sphere = new sphere(vec3(220, 280, 300), 80, new lambertian(pertext));
    d_list[z++] = noise_sphere;

    auto white = new lambertian(vec3(0.73, 0.73, 0.73));
    int num_random_spheres = 1000;
    for (int j = 0; j < num_random_spheres; j++)
    {
        auto random_sphere = new sphere(vec3::random_cuda(&local_rand_state) * 165, 10, white);
        auto rotated_random_sphere = new rotate_y(random_sphere, 15);
        auto translated_random_sphere = new translate(rotated_random_sphere, vec3(-100, 270, 395));
        d_list[z++] = translated_random_sphere;
    }

    // create bvh_nodes
    bvh_node::prefill_nodes(d_bvh_nodes, d_list, list_size);

    *d_camera = camera();
    d_camera->lookfrom = vec3(478, 278, -600);
    d_camera->lookat = vec3(278, 278, 0);
    d_camera->vup = vec3(0, 1, 0);
    d_camera->vfov = 40.0;
    d_camera->image_width = nx;
    d_camera->image_height = ny;
    d_camera->defocus_angle = 0.0;
    d_camera->background = vec3(0, 0, 0);
    d_camera->initialize();
}

void final_scene_wk2(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny)
{
    LOAD_IMAGE_TEXTURE("assets/earthmap.jpg");

    INIT_LIST_AND_TREE(1410);

    create_final_scene_wk2<<<dim3(1, 1), dim3(1, 1)>>>(d_bvh_nodes, d_list, d_camera, d_pixel_data,
                                                       texture.width, texture.height, texture.channels,
                                                       list_size, nx, ny);
}
