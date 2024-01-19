#pragma once

#include "hitable.h"
#include "pdf.h"
#include "ray.h"
#include "texture.h"
#include "vec3.h"

struct scatter_record
{
public:
    vec3 attenuation;
    pdf *pdf_ptr;
    bool skip_pdf;
    ray skip_pdf_ray;
};

class material
{
public:
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const = 0;

    __device__ virtual vec3 emitted(float u, float v, const vec3 &p) const
    {
        return vec3(0, 0, 0);
    }

    __device__ virtual float scattering_pdf(const ray &r_in, const hit_record &rec, const ray &scattered) const = 0;
};

class lambertian : public material
{
public:
    __device__ lambertian(const vec3 &a);
    __device__ lambertian(rtapp::texture *a);
    __device__ ~lambertian();
    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const override;
    __device__ float scattering_pdf(const ray &r_in, const hit_record &rec, const ray &scattered) const override;

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

class diffuse_light : public material
{
public:
    __device__ diffuse_light(rtapp::texture *a) : emit(a) {}
    __device__ diffuse_light(vec3 c);

    __device__ ~diffuse_light();

    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const override;

    __device__ vec3 emitted(float u, float v, const vec3 &p) const override;

    __device__ float scattering_pdf(const ray &r_in, const hit_record &rec, const ray &scattered) const override;

private:
    rtapp::texture *emit;
};

class isotropic : public material
{
public:
    __device__ isotropic(vec3 c) : albedo(new rtapp::solid_color(c)) {}
    __device__ isotropic(rtapp::texture *a) : albedo(a) {}
    __device__ ~isotropic();

    __device__ bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const override;

private:
    rtapp::texture *albedo;
};