#pragma once

#include "aabb.h"
#include "material.h"
#include "vec3.h"

class sphere
{
public:
    __device__ sphere();
    __device__ sphere(vec3 cen, float r, material *m);
    __device__ sphere(vec3 cen1, vec3 cen2, float r, material *m);

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec) const;
    __device__ aabb bounding_box() const;
    __device__ vec3 get_center(float time) const;
    __device__ void set_movable(bool movable);
    __device__ void set_center_vec(vec3 center_vec);

    __device__ ~sphere();

    __device__ static void get_sphere_uv(const vec3 &p, float &u, float &v);

private:
    vec3 center1;
    vec3 center_vec;
    float radius;
    material *mat_ptr;
    bool movable;
    aabb bbox;
};
