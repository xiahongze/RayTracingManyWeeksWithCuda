#pragma once

#include "hitable.h"
#include "ray.h"
#include "texture.h"
#include "vec3.h"

class material
{
public:
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material
{
public:
    __device__ lambertian(const vec3 &a);
    __device__ lambertian(rtapp::texture *a);
    __device__ ~lambertian();
    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const override;

    rtapp::texture *albedo;
};

class metal : public material
{
public:
    __device__ metal(const vec3 &a, float f);
    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const override;

    vec3 albedo;
    float fuzz;
};

class dielectric : public material
{
public:
    __device__ dielectric(float ri);
    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const override;

    float ref_idx;
};
