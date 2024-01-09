#pragma once
#include "ray.h"
#include "vec3.h"

class material;

struct hit_record
{
public:
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
    float u;
    float v;
    bool front_face;

    __device__ void set_face_normal(const ray &r, const vec3 &outward_normal);
};