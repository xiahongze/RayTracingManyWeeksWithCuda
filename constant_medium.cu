#include "constant_medium.h"

__device__ constant_medium::constant_medium(hitable *b, float d, rtapp::texture *a)
    : boundary(b), neg_inv_density(-1 / d), phase_function(new isotropic(a))
{
}

__device__ constant_medium::constant_medium(hitable *b, float d, vec3 c)
    : boundary(b), neg_inv_density(-1 / d), phase_function(new isotropic(c))
{
}

__device__ bool constant_medium::hit(const ray &r, const interval &ray_t, hit_record &rec) const
{
    hit_record rec1, rec2;

    if (!boundary->hit(r, interval::get_universe(), rec1))
        return false;

    if (!boundary->hit(r, interval(rec1.t + 0.0001, FLT_MAX), rec2))
        return false;

    if (rec1.t < ray_t.min)
        rec1.t = ray_t.min;
    if (rec2.t > ray_t.max)
        rec2.t = ray_t.max;

    if (rec1.t >= rec2.t)
        return false;

    if (rec1.t < 0)
        rec1.t = 0;

    auto ray_length = r.direction().length();
    auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    curandState local_rand_state;
    auto hit_distance = neg_inv_density * log(curand_uniform(&local_rand_state));

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.point_at_parameter(rec.t);

    rec.normal = vec3(1, 0, 0); // arbitrary
    rec.front_face = true;      // also arbitrary
    rec.mat_ptr = phase_function;

    return true;
}