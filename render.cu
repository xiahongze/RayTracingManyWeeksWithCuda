#include "material.h"
#include "render.h"
#include "utils.h"
#include "vec3.h"

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 get_ray_color_pixel(const ray &r, bvh_node *d_bvh_nodes, vec3 &backgroound, curandState *local_rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < RAY_MAX_DEPTH; i++)
    {
        hit_record rec;
        if (bvh_node::hit(d_bvh_nodes, cur_ray, interval(0.001f, FLT_MAX), rec))
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
    return backgroound; // exceeded recursion
}

__global__ void render(vec3 *d_fb, int max_x, int max_y, int ns, camera *d_camera, bvh_node *d_bvh_nodes)
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
        col += get_ray_color_pixel(r, d_bvh_nodes, d_camera->background, &local_rand_state);
    }
    col /= float(ns);
    col.to_gamma_space();
    d_fb[pixel_index] = col;
}