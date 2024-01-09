#pragma once

#include "hit_record.h"
#include "ray.h"
#include "texture.h"
#include "vec3.h"

class lambertian
{
public:
    __device__ lambertian(const vec3 &a);
    __device__ lambertian(app_texture *a);
    __device__ ~lambertian();
    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const;

    app_texture *albedo;
};

class metal
{
public:
    __device__ metal(const vec3 &a, float f);
    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const;

    vec3 albedo;
    float fuzz;
};

class dielectric
{
public:
    __device__ dielectric(float ri);
    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const;

    float ref_idx;
};

class diffuse_light
{
public:
    __device__ diffuse_light(app_texture *a) : emit(a) {}
    __device__ diffuse_light(vec3 c);

    __device__ ~diffuse_light();

    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const;

    __device__ vec3 emitted(float u, float v, const vec3 &p) const;

private:
    app_texture *emit;
};

enum material_type
{
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT
};

class material
{
public:
    __device__ material(lambertian *l);
    __device__ material(metal *m);
    __device__ material(dielectric *d);
    __device__ material(diffuse_light *d);
    __device__ ~material();

    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const;
    __device__ vec3 emitted(float u, float v, const vec3 &p) const;

    material_type type;
    lambertian *lamb;
    metal *met;
    dielectric *die;
    diffuse_light *diff;
};