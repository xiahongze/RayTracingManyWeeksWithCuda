#include "material.h"
#include "quad.h"
#include "scenes-wk3.h"
#include "sphere.h"
#include "utils.h"

__global__ void create_final_scene_wk3(bvh_node *d_bvh_nodes, hitable **d_list, hitable_list *d_lights, camera *d_camera,
                                       int list_size, int nx, int ny, int rand_seed)
{
    CHECK_SINGLE_THREAD_BOUNDS();

    auto red = new lambertian(vec3(.65, .05, .05));
    auto white = new lambertian(vec3(.73, .73, .73));
    auto green = new lambertian(vec3(.12, .45, .15));
    auto light = new diffuse_light(vec3(15, 15, 15));

    // Cornell box sides
    d_list[0] = new quad(vec3(555, 0, 0), vec3(0, 0, 555), vec3(0, 555, 0), green);
    d_list[1] = new quad(vec3(0, 0, 555), vec3(0, 0, -555), vec3(0, 555, 0), red);
    d_list[2] = new quad(vec3(0, 555, 0), vec3(555, 0, 0), vec3(0, 0, 555), white);
    d_list[3] = new quad(vec3(0, 0, 555), vec3(555, 0, 0), vec3(0, 0, -555), white);
    d_list[4] = new quad(vec3(555, 0, 555), vec3(-555, 0, 0), vec3(0, 555, 0.1), white);

    // Light
    d_list[5] = new quad(vec3(213, 554, 227), vec3(130, 0, 0), vec3(0, 0, 105), light);

    // Box
    auto box1 = new box(vec3(0, 0, 0), vec3(165, 330, 165), white);
    d_list[6] = new translate(new rotate_y(box1, 15), vec3(265, 0, 295));

    // Glass Sphere
    auto glass = new dielectric(1.5);
    d_list[7] = new sphere(vec3(190, 90, 190), 90, glass);

    // // Light Sources
    // hitable **lights = new hitable *[2];
    // lights[0] = new quad(vec3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), nullptr);
    // lights[1] = new sphere(vec3(190, 90, 190), 90, nullptr);
    // *d_lights = hitable_list(lights, 2);

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

void final_scene_wk3(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, hitable_list *&d_lights, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny, int rand_seed)
{
    INIT_LIST_AND_TREE(8);

    create_final_scene_wk3<<<1, 1>>>(d_bvh_nodes, d_list, d_lights, d_camera, list_size, nx, ny, rand_seed);
}
