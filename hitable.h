#pragma once

#include "aabb.h"
#include "interval.h"
#include "ray.h"

class material;

struct hit_record
{
public:
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

class hitable
{
public:
    __device__ virtual bool hit(const ray &r, const interval ray_t, hit_record &rec) const = 0;

    __device__ virtual aabb bounding_box() const = 0;
};
