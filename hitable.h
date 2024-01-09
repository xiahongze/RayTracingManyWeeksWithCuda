#pragma once

#include "aabb.h"
#include "interval.h"
#include "quad.h"
#include "ray.h"
#include "sphere.h"

class material;

struct hit_record
{
public:
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
    float u;
    float v;
    bool front_face;

    __device__ void set_face_normal(const ray &r, const vec3 &outward_normal);
};

enum class shape_type
{
    SPHERE,
    QUAD,
    BOX
};

class hitable
{
public:
    __device__ hitable(sphere *sphere);
    __device__ hitable(box *box);
    __device__ hitable(quad *quad);
    __device__ ~hitable();

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec) const;
    __device__ aabb bounding_box() const;

private:
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
