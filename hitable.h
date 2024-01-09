#pragma once

#include "aabb.h"
#include "hit_record.h"
#include "interval.h"
#include "quad.h"
#include "ray.h"
#include "sphere.h"

enum class shape_type
{
    SPHERE,
    QUAD,
    BOX
};

class hitable
{
public:
    __device__ hitable(sphere *sphere) : shape(shape_type::SPHERE), sphere(sphere){};
    __device__ hitable(box *box) : shape(shape_type::BOX), box(box){};
    __device__ hitable(quad *quad) : shape(shape_type::QUAD), quad(quad){};
    __device__ ~hitable();

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec) const;
    __device__ aabb bounding_box() const;

    shape_type shape;
    sphere *sphere;
    box *box;
    quad *quad;
};

// class translate : public hitable
// {
// public:
//     __device__ translate(hitable *p, const vec3 &displacement)
//         : object(p), offset(displacement)
//     {
//         bbox = object->bounding_box() + offset;
//     }

//     __device__ ~translate();

//     __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec) const override;

//     __device__ aabb bounding_box() const override;

// private:
//     hitable *object;
//     vec3 offset;
//     aabb bbox;
// };

// class rotate_y : public hitable
// {
// public:
//     __device__ rotate_y(hitable *p, float angle);

//     __device__ ~rotate_y();

//     __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec) const override;

//     __device__ aabb bounding_box() const override { return bbox; }

// private:
//     hitable *object;
//     float sin_theta;
//     float cos_theta;
//     aabb bbox;
// };
