#ifndef HITABLEH
#define HITABLEH

#include "ray.h"
#include "interval.h"

class material;

class hit_record
{
public:
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
    bool front_face;

    __device__ inline void set_face_normal(const ray &r, const vec3 &outward_normal)
    {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hitable
{
public:
    __device__ virtual bool hit(const ray &r, const interval ray_t, hit_record &rec) const = 0;
};

#endif
