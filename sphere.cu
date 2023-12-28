#include "sphere.h"

__device__ sphere::sphere() {}

__device__ sphere::sphere(vec3 cen, float r, material *m) : center1(cen), radius(r), mat_ptr(m), movable(false)
{
    auto rvec = vec3(radius, radius, radius);
    bbox = aabb(center1 - rvec, center1 + rvec);
}

__device__ sphere::sphere(vec3 cen1, vec3 cen2, float r, material *m) : center1(cen1), radius(r), mat_ptr(m), movable(true)
{
    auto rvec = vec3(radius, radius, radius);
    aabb box1(center1 - rvec, center1 + rvec);
    aabb box2(cen2 - rvec, cen2 + rvec);
    bbox = aabb(box1, box2);

    center_vec = cen2 - center1;
}

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

__device__ aabb sphere::bounding_box() const { return bbox; }

__device__ vec3 sphere::get_center(float time) const
{
    // Linearly interpolate from center1 to center2 according to time, where t=0 yields
    // center1, and t=1 yields center2.
    return center1 + time * center_vec;
}

__device__ void sphere::set_movable(bool movable)
{
    this->movable = movable;
}

__device__ void sphere::set_center_vec(vec3 center_vec)
{
    this->center_vec = center_vec;
}

__device__ sphere::~sphere()
{
    delete mat_ptr;
}
