#pragma once
#include "aabb.h"
#include "hitable.h"
#include "material.h"
#include "ray.h"
#include "vec3.h"

class quad : public hitable
{
public:
    __device__ quad(){};

    __device__ quad(const vec3 &_Q, const vec3 &_u, const vec3 &_v, material *m);

    __device__ void set_bounding_box();

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const override;

    __device__ bool is_interior(float a, float b, hit_record &rec) const;

private:
    vec3 Q;
    vec3 u, v;
    material *mat;
    aabb bbox;
    vec3 normal;
    float D;
    vec3 w;
};

class box : public hitable
{
public:
    __device__ box(const vec3 &a, const vec3 &b, material *mat, curandState *local_rand_state);

    __device__ ~box();

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const override;

private:
    quad sides[6];
    aabb bbox;
    material *mat_ptr;
};
