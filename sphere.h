#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include "material.h"

class sphere : public hitable
{
public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m){};
    __device__ virtual bool hit(const ray &r, const interval ray_t, hit_record &rec) const;
    __device__ ~sphere()
    {
        delete mat_ptr;
    }
    vec3 center;
    float radius;
    material *mat_ptr;
};

__device__ bool sphere::hit(const ray &r, const interval ray_t, hit_record &rec) const
{
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

#endif
