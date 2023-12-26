#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ float degrees_to_radians(float degrees)
{
    return degrees * ((float)M_PI) / 180.0f;
}

class camera
{
public:
    int image_width = 1200; // Rendered image width in pixel count
    int image_height = 800; // Rendered image height

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

    __device__ void initialize()
    {
        aspect_ratio = static_cast<float>(image_width) / image_height;
        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * aspect_ratio;

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;   // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v; // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors to the next pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    __device__ ray get_ray(int i, int j, curandState *local_rand_state)
    {
        // Get a randomly-sampled camera ray for the pixel at location i,j, originating from
        // the camera defocus disk.

        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        auto pixel_sample = pixel_center + pixel_sample_square(local_rand_state);

        auto ray_origin = (defocus_angle <= 0.0) ? center : defocus_disk_sample(local_rand_state);
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    __device__ vec3 pixel_sample_square(curandState *local_rand_state) const
    {
        // Returns a random point in the square surrounding a pixel at the origin.
        auto px = -0.5 + curand_uniform(local_rand_state);
        auto py = -0.5 + curand_uniform(local_rand_state);
        return (px * pixel_delta_u) + (py * pixel_delta_v);
    }

    __device__ vec3 pixel_sample_disk(float radius, curandState *local_rand_state) const
    {
        // Generate a sample from the disk of given radius around a pixel at the origin.
        auto p = radius * vec3::random_in_unit_disk(local_rand_state);
        return center + (p.x() * defocus_disk_u) + (p.y() * defocus_disk_v);
    }

    __device__ vec3 defocus_disk_sample(curandState *local_rand_state) const
    {
        // Returns a random point in the camera defocus disk.
        vec3 p = vec3::random_in_unit_disk(local_rand_state);
        return center + p.x() * defocus_disk_u + p.y() * defocus_disk_v;
    }

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

#endif
