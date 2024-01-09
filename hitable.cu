#include "hitable.h"
#include "utils.h"

// __device__ translate::~translate()
// {
//     delete object;
// }

// __device__ bool translate::hit(const ray &r, const interval &ray_t, hit_record &rec) const
// {
//     // Move the ray backwards by the offset
//     ray offset_r(r.origin() - offset, r.direction(), r.get_time());

//     // Determine whether an intersection exists along the offset ray (and if so, where)
//     if (!object->hit(offset_r, ray_t, rec))
//         return false;

//     // Move the intersection point forwards by the offset
//     rec.p += offset;

//     return true;
// }

// __device__ aabb translate::bounding_box() const
// {
//     return bbox;
// }

// __device__ rotate_y::rotate_y(hitable *p, float angle) : object(p)
// {
//     auto radians = degrees_to_radians(angle);
//     sin_theta = sin(radians);
//     cos_theta = cos(radians);
//     bbox = object->bounding_box();

//     vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
//     vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

//     for (int i = 0; i < 2; i++)
//     {
//         for (int j = 0; j < 2; j++)
//         {
//             for (int k = 0; k < 2; k++)
//             {
//                 auto x = i * bbox.x.max + (1 - i) * bbox.x.min;
//                 auto y = j * bbox.y.max + (1 - j) * bbox.y.min;
//                 auto z = k * bbox.z.max + (1 - k) * bbox.z.min;

//                 auto newx = cos_theta * x + sin_theta * z;
//                 auto newz = -sin_theta * x + cos_theta * z;

//                 vec3 tester(newx, y, newz);

//                 for (int c = 0; c < 3; c++)
//                 {
//                     min[c] = fmin(min[c], tester[c]);
//                     max[c] = fmax(max[c], tester[c]);
//                 }
//             }
//         }
//     }

//     bbox = aabb(min, max);
// }

// __device__ rotate_y::~rotate_y()
// {
//     delete object;
// }

// __device__ bool rotate_y::hit(const ray &r, const interval &ray_t, hit_record &rec) const
// {
//     // Change the ray from world space to object space
//     auto origin = r.origin();
//     auto direction = r.direction();

//     origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
//     origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

//     direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
//     direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

//     ray rotated_r(origin, direction, r.get_time());

//     // Determine whether an intersection exists in object space (and if so, where)
//     if (!object->hit(rotated_r, ray_t, rec))
//         return false;

//     // Change the intersection point from object space to world space
//     auto p = rec.p;
//     p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
//     p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

//     // Change the normal from object space to world space
//     auto normal = rec.normal;
//     normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
//     normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

//     rec.p = p;
//     rec.normal = normal;

//     return true;
// }

// __device__ hitable::hitable(sphere *sphere)
// {
//     this->shape = shape_type::SPHERE;
//     this->sphere = sphere;
// }

// __device__ hitable::hitable(box *box)
// {
//     this->shape = shape_type::BOX;
//     this->box = box;
// }

// __device__ hitable::hitable(quad *quad)
// {
//     this->shape = shape_type::QUAD;
//     this->quad = quad;
// }

__device__ hitable::~hitable()
{
    switch (shape)
    {
    case shape_type::SPHERE:
        delete sphere;
        break;
    case shape_type::BOX:
        delete box;
        break;
    case shape_type::QUAD:
        delete quad;
        break;
    }
}

__device__ bool hitable::hit(const ray &r, const interval &ray_t, hit_record &rec) const
{
    switch (shape)
    {
    case shape_type::SPHERE:
        return sphere->hit(r, ray_t, rec);
    case shape_type::BOX:
        return box->hit(r, ray_t, rec);
    case shape_type::QUAD:
        return quad->hit(r, ray_t, rec);
    default:
        return false;
    }
}

__device__ aabb hitable::bounding_box() const
{
    switch (shape)
    {
    case shape_type::SPHERE:
        return sphere->bounding_box();
    case shape_type::BOX:
        return box->bounding_box();
    case shape_type::QUAD:
        return quad->bounding_box();
    default:
        return aabb();
    }
}
