#pragma once

#include "ray.h"
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ float degrees_to_radians(float degrees);

class camera
{
public:
    int image_width = 1200;                   // Rendered image width in pixel count
    int image_height = 800;                   // Rendered image height
    vec3 background = vec3(0.70, 0.80, 1.00); // Scene background color

    float vfov = 90;                // Vertical view angle (field of view)
    vec3 lookfrom = vec3(0, 0, -1); // Point camera is looking from
    vec3 lookat = vec3(0, 0, 0);    // Point camera is looking at
    vec3 vup = vec3(0, 1, 0);       // Camera-relative "up" direction

    float defocus_angle = 0; // Variation angle of rays through each pixel
    float focus_dist = 10;   // Distance from camera lookfrom point to plane of perfect focus

    __device__ camera() {}

    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, int image_width, int image_height, float focus_dist, float defocus_angle)
        : lookfrom(lookfrom), lookat(lookat), vup(vup), vfov(vfov), image_width(image_width), image_height(image_height), focus_dist(focus_dist), defocus_angle(defocus_angle)
    {
        initialize();
    }

    __device__ void initialize();

    __device__ ray get_ray(int i, int j, int s_i, int s_j, int sqrt_spp, curandState *local_rand_state) const;

    __device__ vec3 pixel_sample_square(int s_i, int s_j, int sqrt_spp, curandState *local_rand_state) const;

    __device__ vec3 pixel_sample_disk(float radius, curandState *local_rand_state) const;

    __device__ vec3 defocus_disk_sample(curandState *local_rand_state) const;

private:
    float aspect_ratio = 1.0; // Ratio of image width over height
    vec3 center;              // Camera center
    vec3 pixel00_loc;         // Location of pixel 0, 0
    vec3 pixel_delta_u;       // Offset to pixel to the right
    vec3 pixel_delta_v;       // Offset to pixel below
    vec3 u, v, w;             // Camera frame basis vectors
    vec3 defocus_disk_u;      // Defocus disk horizontal radius
    vec3 defocus_disk_v;      // Defocus disk vertical radius
};
