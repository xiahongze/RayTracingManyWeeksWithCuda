#include "quad.h"

__device__ quad::quad(const vec3 &_Q, const vec3 &_u, const vec3 &_v, material *m)
    : Q(_Q), u(_u), v(_v), mat(m)
{
    auto n = cross(u, v);
    normal = unit_vector(n);
    D = dot(normal, Q);
    w = n / dot(n, n);

    set_bounding_box();
}

__device__ void quad::set_bounding_box()
{
    bbox = aabb(Q, Q + u + v).pad();
}

__device__ bool quad::hit(const ray &r, const interval &ray_t, hit_record &rec) const
{
    auto denom = dot(normal, r.direction());

    // No hit if the ray is parallel to the plane.
    if (fabs(denom) < 1e-8)
        return false;

    // Return false if the hit point parameter t is outside the ray interval.
    auto t = (D - dot(normal, r.origin())) / denom;
    if (!ray_t.contains(t))
        return false;

    // Determine the hit point lies within the planar shape using its plane coordinates.
    auto intersection = r.point_at_parameter(t);
    vec3 planar_hitpt_vector = intersection - Q;
    auto alpha = dot(w, cross(planar_hitpt_vector, v));
    auto beta = dot(w, cross(u, planar_hitpt_vector));

    if (!is_interior(alpha, beta, rec))
        return false;

    // Ray hits the 2D shape; set the rest of the hit record and return true.
    rec.t = t;
    rec.p = intersection;
    rec.mat_ptr = mat;
    rec.set_face_normal(r, normal);

    return true;
}

__device__ bool quad::is_interior(float a, float b, hit_record &rec) const
{
    // Given the hit point in plane coordinates, return false if it is outside the
    // primitive, otherwise set the hit record UV coordinates and return true.

    if ((a < 0) || (1 < a) || (b < 0) || (1 < b))
        return false;

    rec.u = a;
    rec.v = b;
    return true;
}

__device__ void quad::box(const vec3 &a, const vec3 &b, material *mat, hitable **hitable_list, int start)
{
    // Construct the two opposite vertices with the minimum and maximum coordinates.
    auto min = vec3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
    auto max = vec3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

    auto dx = vec3(max.x() - min.x(), 0, 0);
    auto dy = vec3(0, max.y() - min.y(), 0);
    auto dz = vec3(0, 0, max.z() - min.z());

    hitable_list[start] = new quad(vec3(min.x(), min.y(), max.z()), dx, dy, mat);      // front
    hitable_list[start + 1] = new quad(vec3(max.x(), min.y(), max.z()), -dz, dy, mat); // right
    hitable_list[start + 2] = new quad(vec3(max.x(), min.y(), min.z()), -dx, dy, mat); // back
    hitable_list[start + 3] = new quad(vec3(min.x(), min.y(), min.z()), dz, dy, mat);  // left
    hitable_list[start + 4] = new quad(vec3(min.x(), max.y(), max.z()), dx, -dz, mat); // top
    hitable_list[start + 5] = new quad(vec3(min.x(), min.y(), min.z()), dx, dz, mat);  // bottom
}