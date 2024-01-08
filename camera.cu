#include "camera.h"
#include "utils.h"

__device__ void camera::initialize()
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

__device__ ray camera::get_ray(int i, int j, curandState *local_rand_state) const
{
    // Get a randomly-sampled camera ray for the pixel at location i,j, originating from
    // the camera defocus disk.

    auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
    auto pixel_sample = pixel_center + pixel_sample_square(local_rand_state);

    auto ray_origin = (defocus_angle <= 0.0) ? center : defocus_disk_sample(local_rand_state);
    auto ray_direction = pixel_sample - ray_origin;
    auto ray_time = curand_uniform(local_rand_state);

    return ray(ray_origin, ray_direction, ray_time);
}

__device__ vec3 camera::pixel_sample_square(curandState *local_rand_state) const
{
    // Returns a random point in the square surrounding a pixel at the origin.
    auto px = -0.5 + curand_uniform(local_rand_state);
    auto py = -0.5 + curand_uniform(local_rand_state);
    return (px * pixel_delta_u) + (py * pixel_delta_v);
}

__device__ vec3 camera::pixel_sample_disk(float radius, curandState *local_rand_state) const
{
    // Generate a sample from the disk of given radius around a pixel at the origin.
    auto p = radius * vec3::random_in_unit_disk(local_rand_state);
    return center + (p.x() * defocus_disk_u) + (p.y() * defocus_disk_v);
}

__device__ vec3 camera::defocus_disk_sample(curandState *local_rand_state) const
{
    // Returns a random point in the camera defocus disk.
    vec3 p = vec3::random_in_unit_disk(local_rand_state);
    return center + p.x() * defocus_disk_u + p.y() * defocus_disk_v;
}