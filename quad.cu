#include "quad.h"
#include "utils.h"

__device__ quad::quad(const vec3 &_Q, const vec3 &_u, const vec3 &_v, material *m)
    : Q(_Q), u(_u), v(_v), mat(m)
{
    auto n = cross(u, v);
    normal = unit_vector(n);
    D = dot(normal, Q);
    w = n / dot(n, n);

    area = n.length();

    set_bounding_box();
}

__device__ void quad::set_bounding_box()
{
    bbox = aabb(Q, Q + u + v).pad();
}

__device__ bool quad::hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const
{
    auto denom = dot(normal, r.direction());

    // No hit if the ray is parallel to the plane.
    if (fabs(denom) < 1e-4)
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

__device__ float quad::pdf_value(const vec3 &origin, const vec3 &v, curandState *local_rand_state) const
{
    hit_record rec;
    if (!this->hit(ray(origin, v), interval(0.001, FLT_MAX), rec, local_rand_state))
        return 0;

    auto distance_squared = rec.t * rec.t * v.squared_length();
    auto cosine = fabs(dot(v, rec.normal) / v.length());

    return distance_squared / (cosine * area);
}

__device__ vec3 quad::random(const vec3 &origin, curandState *local_rand_state) const
{
    auto p = Q + (curand_uniform(local_rand_state) * u) + (curand_uniform(local_rand_state) * v);
    return p - origin;
}

__device__ box::box(const vec3 &a, const vec3 &b, material *mat)
{
    // Construct the two opposite vertices with the minimum and maximum coordinates.
    auto min = vec3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
    auto max = vec3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

    // need to perturb the box a little bit to avoid degenerate quads
    // if not, you may see black side if camera is facing the box perpendicularly
    auto dx = vec3(max.x() - min.x(), 0.1, 0);
    auto dy = vec3(0, max.y() - min.y(), 0.1);
    auto dz = vec3(0.1, 0, max.z() - min.z());

    sides[0] = quad(vec3(min.x(), min.y(), max.z()), dx, dy, mat);  // front
    sides[1] = quad(vec3(max.x(), min.y(), max.z()), -dz, dy, mat); // right
    sides[2] = quad(vec3(max.x(), min.y(), min.z()), -dx, dy, mat); // back
    sides[3] = quad(vec3(min.x(), min.y(), min.z()), dz, dy, mat);  // left
    sides[4] = quad(vec3(min.x(), max.y(), max.z()), dx, -dz, mat); // top
    sides[5] = quad(vec3(min.x(), min.y(), min.z()), dx, dz, mat);  // bottom

    bbox = aabb(min, max);
}

__device__ box::~box()
{
    delete[] sides;
    delete mat_ptr;
}

__device__ bool box::hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;

    for (int i = 0; i < 6; i++)
    {
        if (sides[i].hit(r, interval(ray_t.min, closest_so_far), temp_rec, local_rand_state))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}
