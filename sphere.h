#pragma once

#include "aabb.h"
#include "hitable.h"
#include "material.h"
#include "vec3.h"

class sphere : public hitable
{
public:
    __device__ sphere();
    __device__ sphere(vec3 cen, float r, material *m);
    __device__ sphere(vec3 cen1, vec3 cen2, float r, material *m);

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const override;
    __device__ aabb bounding_box() const override;
    __device__ float pdf_value(const vec3 &o, const vec3 &v, curandState *local_rand_state) const override;
    __device__ vec3 random(const vec3 &o, curandState *local_rand_state) const override;

    __device__ vec3 get_center(float time) const;
    __device__ void set_movable(bool movable);
    __device__ void set_center_vec(vec3 center_vec);

    __device__ ~sphere();

    __device__ static void get_sphere_uv(const vec3 &p, float &u, float &v);
    __device__ static vec3 random_to_sphere(float radius, float distance_squared, curandState *local_rand_state);

private:
    vec3 center1;
    vec3 center_vec;
    float radius;
    material *mat_ptr;
    bool movable;
    aabb bbox;
};
