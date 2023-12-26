#pragma once

struct hit_record;

#include "ray.h"

__device__ float schlick(float cosine, float ref_idx)
{
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const vec3 &v, const vec3 &n, float ni_over_nt, vec3 &refracted)
{
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
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

        // determine whether ray is entering or leaving the material
        // and calculate the refractive index accordingly

        // assume facing outwards by default
        vec3 outward_normal = rec.normal;
        float ni_over_nt = 1.0f / ref_idx;
        float cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();

        if (dot(r_in.direction(), rec.normal) > 0.0f)
        {
            // facing inwards
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
        }

        vec3 refracted;
        bool can_refract = refract(r_in.direction(), outward_normal, ni_over_nt, refracted);
        if (can_refract && curand_uniform(local_rand_state) > schlick(cosine, ref_idx))
        {
            scattered = ray(rec.p, refracted);
        }
        else
        {
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            scattered = ray(rec.p, reflected);
        }

        return true;
    }

    float ref_idx;
};
