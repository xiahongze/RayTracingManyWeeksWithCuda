#include "bvh.h"
#include "camera.h"
#include "material.h"
#include "sphere.h"
#include "texture.h"
#include "utils.h"

__global__ void
create_random_spheres(bvh_node *d_bvh_nodes, hitable **d_list, camera *d_camera, int list_size, int nx, int ny, bool bounce, float bounce_pct, bool checkered)
{
    CHECK_SINGLE_THREAD_BOUNDS();

    for (int i = 0; i < 22; i++)
    {
        for (int j = 0; j < 22; j++)
        {
            int idx = 4 + j * 22 + i;

            curandState local_rand_state;
            curand_init(RAND_SEED + idx, 0, 0, &local_rand_state);

            int a = i - 11;
            int b = j - 11;

            float radius = 0.2;
            vec3 center(a + RND, radius, b + RND);

            float choose_mat = RND;
            if (choose_mat < 0.8f)
            {
                d_list[idx] = new hitable(new sphere(center, radius, new material(new lambertian(vec3::random_cuda(&local_rand_state).as_squared()))));
            }
            else if (choose_mat < 0.95f)
            {
                d_list[idx] = new hitable(new sphere(center, radius,
                                                     new material(new metal(1.0f + -0.5f * vec3::random_cuda(&local_rand_state), 0.5f * RND))));
            }
            else
            {
                d_list[idx] = new hitable(new sphere(center, radius, new material(new dielectric(1.5))));
            }

            if (bounce && RND < bounce_pct) // only 1/3 are allowed to move
            {
                d_list[idx]->sphere->set_movable(true);
                d_list[idx]->sphere->set_center_vec(vec3(0, RND * radius * 2, 0));
            }
        }
    }

    // if (checkered)
    // {
    //     d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
    //                            new lambertian(new checker_texture(
    //                                0.8, vec3(0.2, 0.3, 0.1), vec3(0.9, 0.9, 0.9))));
    // }
    // else
    {

        d_list[0] = new hitable(new sphere(vec3(0, -1000.0, -1), 1000,
                                           new material(new lambertian(vec3(0.5, 0.5, 0.5)))));
    }
    d_list[1] = new hitable(new sphere(vec3(0, 1, 0), 1.0, new material(new dielectric(1.5))));
    d_list[2] = new hitable(new sphere(vec3(-4, 1, 0), 1.0, new material(new lambertian(vec3(0.4, 0.2, 0.1)))));
    d_list[3] = new hitable(new sphere(vec3(4, 1, 0), 1.0, new material(new metal(vec3(0.7, 0.6, 0.5), 0.0))));

    // create bvh_nodes
    bvh_node::prefill_nodes(d_bvh_nodes, d_list, list_size);

    *d_camera = camera();
    d_camera->lookfrom = vec3(13, 2, 3);
    d_camera->lookat = vec3(0, 0, 0);
    d_camera->vup = vec3(0, 1, 0);
    d_camera->vfov = 30.0;
    d_camera->image_width = nx;
    d_camera->image_height = ny;
    d_camera->defocus_angle = 0.6;
    d_camera->initialize();
}

void random_spheres(bvh_node *&h_bvh_nodes, bvh_node *&d_bvh_nodes, hitable **&d_list, camera *&d_camera, int &list_size, int &tree_size, int nx, int ny, bool bounce, float bounce_pct, bool checkered)
{
    INIT_LIST_AND_TREE(22 * 22 + 1 + 3);

    create_random_spheres<<<dim3(1, 1), dim3(1, 1)>>>(d_bvh_nodes, d_list, d_camera, list_size, nx, ny, bounce, bounce_pct, checkered);
}