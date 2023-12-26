#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "interval.h"

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
__device__ vec3 get_ray_color_pixel(const ray &r, hitable **world, curandState *local_rand_state)
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

__global__ void render(vec3 *d_fb, int max_x, int max_y, int ns, camera *d_camera, hitable **d_world)
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

__global__ void create_world(hitable_list **d_world, hitable **d_list, camera *d_camera, int list_size, int nx, int ny)
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

    vec3 center(a + RND, 0.2, b + RND);
    float choose_mat = RND;
    if (choose_mat < 0.8f)
    {
        d_list[idx] = new sphere(center, 0.2, new lambertian(vec3::random_cuda(&local_rand_state).as_squared()));
    }
    else if (choose_mat < 0.95f)
    {
        d_list[idx] = new sphere(center, 0.2,
                                 new metal(1.0f + 0.5f * (vec3::random_cuda(&local_rand_state)), 0.5f * RND));
    }
    else
    {
        d_list[idx] = new sphere(center, 0.2, new dielectric(1.5));
    }

    if (i == 0 && j == 0)
    {
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        d_list[1] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[2] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[3] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
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

int main()
{
    int nx = 1200;
    int ny = 800;
    int ns = 10;
    int tx = 6;
    int ty = 4;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // make our world of hitables & the camera
    hitable_list **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable_list *)));
    hitable **d_list;
    int list_size = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&d_list, list_size * sizeof(hitable *)));
    camera *d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera)));
    create_world<<<dim3(22, 22), dim3(1, 1)>>>(d_world, d_list, d_camera, list_size, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();

    // Render our buffer
    dim3 blocks(nx / tx + (nx % tx ? 1 : 0), ny / ty + (ny % ty ? 1 : 0));
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, (hitable **)d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image, allocated with cudaMallocManaged can be directly accessed on host
    std::cout << "P3\n"
              << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    free_world<<<1, 1>>>(d_world);
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
