#include "hitable.h"
#include "utils.h"

__device__ void hit_record::set_face_normal(const ray &r, const vec3 &outward_normal)
{
    // Sets the hit record normal vector.
    // NOTE: the parameter `outward_normal` is assumed to have unit length.

    front_face = dot(r.direction(), outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
}

__device__ translate::~translate()
{
    delete object;
}

__device__ bool translate::hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const
{
    // Move the ray backwards by the offset
    ray offset_r(r.origin() - offset, r.direction(), r.get_time());

    // Determine whether an intersection exists along the offset ray (and if so, where)
    if (!object->hit(offset_r, ray_t, rec, local_rand_state))
        return false;

    // Move the intersection point forwards by the offset
    rec.p += offset;

    return true;
}

__device__ aabb translate::bounding_box() const
{
    return bbox;
}

__device__ rotate_y::rotate_y(hitable *p, float angle) : object(p)
{
    auto radians = degrees_to_radians(angle);
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    bbox = object->bounding_box();

    vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                auto x = i * bbox.x.max + (1 - i) * bbox.x.min;
                auto y = j * bbox.y.max + (1 - j) * bbox.y.min;
                auto z = k * bbox.z.max + (1 - k) * bbox.z.min;

                auto newx = cos_theta * x + sin_theta * z;
                auto newz = -sin_theta * x + cos_theta * z;

                vec3 tester(newx, y, newz);

                for (int c = 0; c < 3; c++)
                {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }

    bbox = aabb(min, max);
}

__device__ rotate_y::~rotate_y()
{
    delete object;
}

__device__ bool rotate_y::hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const
{
    // Change the ray from world space to object space
    auto origin = r.origin();
    auto direction = r.direction();

    origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
    origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

    direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
    direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

    ray rotated_r(origin, direction, r.get_time());

    // Determine whether an intersection exists in object space (and if so, where)
    if (!object->hit(rotated_r, ray_t, rec, local_rand_state))
        return false;

    // Change the intersection point from object space to world space
    auto p = rec.p;
    p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
    p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

    // Change the normal from object space to world space
    auto normal = rec.normal;
    normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
    normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

    rec.p = p;
    rec.normal = normal;

    return true;
}

__device__ hitable_list::hitable_list(hitable **l, int n) : list(l), list_size(n)
{
    for (int i = 0; i < n; ++i)
    {
        bbox = aabb(bbox, list[i]->bounding_box());
    }
}

__device__ hitable_list::~hitable_list()
{
    for (int i = 0; i < list_size; ++i)
    {
        delete list[i];
    }
    delete[] list;
}

__device__ bool hitable_list::hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;

    for (int i = 0; i < list_size; ++i)
    {
        if (list[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec, local_rand_state))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

__device__ aabb hitable_list::bounding_box() const
{
    return bbox;
}

__device__ float hitable_list::pdf_value(const vec3 &o, const vec3 &v, curandState *local_rand_state) const
{
    auto weight = 1.0f / list_size;
    auto sum = 0.0f;

    for (int i = 0; i < list_size; i++)
    {
        sum += weight * list[i]->pdf_value(o, v, local_rand_state);
    }

    return sum;
}

__device__ vec3 hitable_list::random(const vec3 &o, curandState *local_rand_state) const
{
    int index = int(curand_uniform(local_rand_state) * list_size);
    return list[index]->random(o, local_rand_state);
}
