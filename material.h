#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"
#include "hitable.h"

__device__ inline float fmin_dev(float a, float b)
{
    return a < b ? a : b;
}

__device__ float reflectance(float cosine, float ref_idx)
{
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ vec3 refract(const vec3 &uv, const vec3 &n, float etai_over_etat)
{
    auto cos_theta = fmin_dev(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2.0f * dot(v, n) * n;
}

class material
{
public:
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material
{
public:
    __device__ lambertian(const vec3 &a) : albedo(a) {}
    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const
    {
        vec3 scatter_direction = rec.normal + vec3::random_in_unit_sphere(local_rand_state);

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

    vec3 albedo;
};

class metal : public material
{
public:
    __device__ metal(const vec3 &a, float f) : albedo(a)
    {
        if (f < 1)
            fuzz = f;
        else
            fuzz = 1;
    }
    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * vec3::random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
    vec3 albedo;
    float fuzz;
};

class dielectric : public material
{
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ bool scatter(const ray &r_in,
                            const hit_record &rec,
                            vec3 &attenuation,
                            ray &scattered,
                            curandState *local_rand_state) const
    {
        attenuation = vec3(1.0, 1.0, 1.0);
        float refraction_ratio = rec.front_face ? (1.0 / ref_idx) : ref_idx;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fmin_dev(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

    float ref_idx;
};
#endif
