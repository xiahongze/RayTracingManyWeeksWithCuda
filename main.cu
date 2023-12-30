#include "bvh.h"
#include "camera.h"
#include "cmd_parser.h"
#include "hitable_list.h"
#include "image_utils.h"
#include "interval.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <time.h>

#ifndef RAY_MAX_DEPTH
#define RAY_MAX_DEPTH 50
#endif

#ifndef RAND_SEED
#define RAND_SEED 1984
#endif

#define RND (curand_uniform(&local_rand_state))

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 get_ray_color_pixel(const ray &r, hitable_list **world, curandState *local_rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < RAY_MAX_DEPTH; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, interval(0.001f, FLT_MAX), rec))
        {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render(vec3 *d_fb, int max_x, int max_y, int ns, camera *d_camera, hitable_list **d_world)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;

    curandState local_rand_state;
    curand_init(RAND_SEED + pixel_index, 0, 0, &local_rand_state);

    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++)
    {
        ray r = d_camera->get_ray(i, j, &local_rand_state);
        col += get_ray_color_pixel(r, d_world, &local_rand_state);
    }
    col /= float(ns);
    col.to_gamma_space();
    d_fb[pixel_index] = col;
}

__global__ void create_world(bvh_node *d_bvh_nodes, hitable_list **d_world, hitable **d_list, camera *d_camera, int list_size, int nx, int ny, bool bounce, float bounce_pct)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= 22) || (j >= 22))
        return;
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
        d_list[idx] = new sphere(center, radius, new lambertian(vec3::random_cuda(&local_rand_state).as_squared()));
    }
    else if (choose_mat < 0.95f)
    {
        d_list[idx] = new sphere(center, radius,
                                 new metal(1.0f + 0.5f * (vec3::random_cuda(&local_rand_state)), 0.5f * RND));
    }
    else
    {
        d_list[idx] = new sphere(center, radius, new dielectric(1.5));
    }

    if (bounce && RND < bounce_pct) // only 1/3 are allowed to move
    {
        ((sphere *)d_list[idx])->set_movable(true);
        ((sphere *)d_list[idx])->set_center_vec(vec3(0, RND * radius * 2, 0));
    }

    if (i == 0 && j == 0)
    {
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        d_list[1] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[2] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[3] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

        // create bvh_nodes
        bvh_node::prefill_nodes(d_bvh_nodes, d_list, list_size);

        *d_world = new hitable_list(d_list, list_size);

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
}

__global__ void free_world(hitable_list **d_world)
{
    if (threadIdx.x > 0 || blockIdx.x > 0)
        return;
    delete *d_world;
}

int main(int argc, char **argv)
{
    auto cmd_opts = parse_command_line(argc, argv);

    int num_pixels = cmd_opts.image_width * cmd_opts.image_height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // make our world of hitables & the camera
    hitable_list **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable_list *)));
    hitable **d_list;
    int list_size = 22 * 22 + 1 + 3;

    // create two arrays of bvh_nodes on host and device
    bvh_node *d_bvh_nodes;
    checkCudaErrors(cudaMalloc((void **)&d_bvh_nodes, 2 * list_size * sizeof(bvh_node)));
    bvh_node *h_bvh_nodes = new bvh_node[list_size * 2]; // binary tree

    checkCudaErrors(cudaMalloc((void **)&d_list, list_size * sizeof(hitable *)));
    camera *d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera)));
    create_world<<<dim3(1, 1), dim3(22, 22)>>>(d_bvh_nodes, d_world, d_list, d_camera, list_size,
                                               cmd_opts.image_width, cmd_opts.image_height, cmd_opts.bounce, cmd_opts.bounce_pct);
    checkCudaErrors(cudaGetLastError());
    // copy bvh_nodes from device to host
    checkCudaErrors(cudaMemcpy(h_bvh_nodes, d_bvh_nodes, list_size * sizeof(bvh_node), cudaMemcpyDeviceToHost));
    std::cout << "bvh_nodes copied to host\n";
    // build bvh tree on host
    int tree_height = build_tree(h_bvh_nodes, list_size);

    clock_t start, stop;
    start = clock();

    // Render our buffer
    dim3 blocks(cmd_opts.image_width / cmd_opts.tx + (cmd_opts.image_width % cmd_opts.tx ? 1 : 0),
                cmd_opts.image_height / cmd_opts.ty + (cmd_opts.image_height % cmd_opts.ty ? 1 : 0));
    dim3 threads(cmd_opts.tx, cmd_opts.ty);
    render<<<blocks, threads>>>(fb, cmd_opts.image_width, cmd_opts.image_height, cmd_opts.samples_per_pixel, d_camera, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image, allocated with cudaMallocManaged can be directly accessed on host
    writeJPGImage(cmd_opts.output_file.c_str(), cmd_opts.image_width, cmd_opts.image_height, fb);

    // clean up
    free_world<<<1, 1>>>(d_world);
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_bvh_nodes));
    checkCudaErrors(cudaFree(fb));
    delete[] h_bvh_nodes;

    cudaDeviceReset();
}
