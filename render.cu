#include "material.h"
#include "render.h"
#include "utils.h"
#include "vec3.h"
#include <curand_kernel.h>

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 get_ray_color_pixel(const int max_depth, const ray &r, bvh_node *d_bvh_nodes, vec3 &backgroound, curandState *local_rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    vec3 final_color(0, 0, 0);
    for (int i = 0; i < max_depth; i++)
    {
        hit_record rec;
        if (!bvh_node::hit(d_bvh_nodes, cur_ray, interval(0.001f, FLT_MAX), rec, local_rand_state))
        {
            final_color += backgroound * cur_attenuation;
            break;
        }

        scatter_record srec;

        vec3 color_from_emission = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);

        if (!rec.mat_ptr->scatter(cur_ray, rec, srec, local_rand_state))
        {
            final_color += color_from_emission * cur_attenuation;
            break;
        }

        if (srec.skip_pdf)
        {
            cur_attenuation *= srec.attenuation;
            cur_ray = srec.skip_pdf_ray;
            continue;
        }

        // // do sample
        // auto p = srec.pdf_ptr;
        // ray scattered = ray(rec.p, p->generate(local_rand_state), r.get_time());
        // auto pdf_val = p->value(scattered.direction(), local_rand_state);

        // float scattering_pdf = rec.mat_ptr->scattering_pdf(r, rec, scattered);

        // cur_ray = scattered;
        // cur_attenuation *= srec.attenuation * scattering_pdf / pdf_val;
    }
    return final_color; // exceeded recursion
}

__global__ void render(vec3 *d_fb, int max_x, int max_y, int ns, int max_depth, int rand_seed, camera *d_camera, bvh_node *d_bvh_nodes)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;

    int sqrt_spp = static_cast<int>(sqrt(ns));
    float recip_sqrt_spp = 1.0 / sqrt_spp;

    curandState local_rand_state;
    curand_init(rand_seed + pixel_index, 0, 0, &local_rand_state);

    vec3 col(0, 0, 0);
    for (int s_j = 0; s_j < sqrt_spp; ++s_j)
    {
        for (int s_i = 0; s_i < sqrt_spp; ++s_i)
        {
            ray r = d_camera->get_ray(i, j, s_i, s_j, recip_sqrt_spp, &local_rand_state);
            // can call vec3.clamp() here but not here because it help with debugging purpose
            col += get_ray_color_pixel(max_depth, r, d_bvh_nodes, d_camera->background, &local_rand_state);
        }
    }
    col /= float(ns);
    col.to_gamma_space();
    d_fb[pixel_index] = col;
}