#pragma once

#include "hitable.h"
#include "material.h"

class sphere : public hitable
{
public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, float r, material *m) : center1(cen), radius(r), mat_ptr(m), movable(false){};
    __device__ sphere(vec3 cen1, vec3 cen2, float r, material *m) : center1(cen1), radius(r), mat_ptr(m), movable(true), center_vec(){};

    __device__ bool hit(const ray &r, const interval ray_t, hit_record &rec) const override;
    __device__ vec3 get_center(float time) const;
    __device__ void set_movable(bool is_moving)
    {
        this->movable = is_moving;
    }
    __device__ void set_center_vec(vec3 center_vec)
    {
        this->center_vec = center_vec;
    }

    __device__ ~sphere()
    {
        delete mat_ptr;
    }

private:
    vec3 center1;
    vec3 center_vec;
    float radius;
    material *mat_ptr;
    bool movable;
};

__device__ bool sphere::hit(const ray &r, const interval ray_t, hit_record &rec) const
{
    vec3 center = movable ? get_center(r.get_time()) : center1;
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float half_b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0)
        return false;

    // Find the nearest root that lies in the acceptable range.
    float sqrtd = sqrt(discriminant);
    float root = (-half_b - sqrtd) / a;
    if (!ray_t.surrounds(root))
    {
        root = (-half_b + sqrtd) / a;
        if (!ray_t.surrounds(root))
            return false;
    }

    rec.t = root;
    rec.p = r.point_at_parameter(rec.t);
    rec.normal = (rec.p - center) / radius;
    rec.mat_ptr = mat_ptr;

    return true;
}

__device__ vec3 sphere::get_center(float time) const
{
    // Linearly interpolate from center1 to center2 according to time, where t=0 yields
    // center1, and t=1 yields center2.
    return center1 + time * center_vec;
}