#pragma once
#include "aabb.h"
#include "hitable.h"
#include "material.h"
#include "ray.h"
#include "vec3.h"

class quad : public hitable
{
public:
    __device__ quad(const vec3 &_Q, const vec3 &_u, const vec3 &_v, material *m);

    __device__ void set_bounding_box();

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec) const override;

    __device__ bool is_interior(float a, float b, hit_record &rec) const;

    __device__ static void box(const vec3 &a, const vec3 &b, material *mat, hitable **hitable_list, int start);

private:
    vec3 Q;
    vec3 u, v;
    material *mat;
    aabb bbox;
    vec3 normal;
    float D;
    vec3 w;
};