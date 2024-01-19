#include "onb.h"
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

__device__ bool sphere::hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const
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
    rec.set_face_normal(r, rec.normal);
    rec.mat_ptr = mat_ptr;
    get_sphere_uv(rec.normal, rec.u, rec.v);

    return true;
}

__device__ aabb sphere::bounding_box() const { return bbox; }

__device__ float sphere::pdf_value(const vec3 &o, const vec3 &v, curandState *local_rand_state) const
{
    // This method only works for stationary spheres.

    hit_record rec;
    if (!this->hit(ray(o, v), interval(0.001, FLT_MAX), rec, local_rand_state))
        return 0;

    auto cos_theta_max = sqrt(1 - radius * radius / (center1 - o).squared_length());
    auto solid_angle = 2 * M_PI * (1 - cos_theta_max);

    return 1 / solid_angle;
}

__device__ vec3 sphere::random(const vec3 &o, curandState *local_rand_state) const
{
    vec3 direction = center1 - o;
    auto distance_squared = direction.squared_length();
    onb uvw;
    uvw.build_from_w(direction);
    return uvw.local(random_to_sphere(radius, distance_squared, local_rand_state));
}

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

__device__ void sphere::get_sphere_uv(const vec3 &p, float &u, float &v)
{
    // p: a given point on the sphere of radius one, centered at the origin.
    // u: returned value [0,1] of angle around the Y axis from X=-1.
    // v: returned value [0,1] of angle from Y=-1 to Y=+1.
    //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
    //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
    //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

    auto theta = acos(-p.y());
    auto phi = atan2(-p.z(), p.x()) + M_PI;

    u = phi / (2 * M_PI);
    v = theta / M_PI;
}

__device__ vec3 sphere::random_to_sphere(float radius, float distance_squared, curandState *local_rand_state)
{
    auto r1 = curand_normal(local_rand_state);
    auto r2 = curand_normal(local_rand_state);
    auto z = 1 + r2 * (sqrt(1 - radius * radius / distance_squared) - 1);

    auto phi = 2 * M_PI * r1;
    auto x = cos(phi) * sqrt(1 - z * z);
    auto y = sin(phi) * sqrt(1 - z * z);

    return vec3(x, y, z);
}
