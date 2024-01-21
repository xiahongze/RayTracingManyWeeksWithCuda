#pragma once

#include "aabb.h"
#include "interval.h"
#include "ray.h"

class material;

struct hit_record
{
public:
    vec3 p;
    vec3 normal;
    material *mat_ptr;
    float t;
    float u;
    float v;
    bool front_face;

    __device__ void set_face_normal(const ray &r, const vec3 &outward_normal);
};

class hitable
{
public:
    __device__ virtual bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const = 0;

    __device__ virtual aabb bounding_box() const = 0;

    __device__ virtual float pdf_value(const vec3 &o, const vec3 &v, curandState *local_rand_state) const
    {
        return 0.0;
    }

    __device__ virtual vec3 random(const vec3 &o, curandState *local_rand_state) const
    {
        return vec3(1, 0, 0);
    }
};

class translate : public hitable
{
public:
    __device__ translate(hitable *p, const vec3 &displacement)
        : object(p), offset(displacement)
    {
        bbox = object->bounding_box() + offset;
    }

    __device__ ~translate();

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const override;

    __device__ aabb bounding_box() const override;

private:
    hitable *object;
    vec3 offset;
    aabb bbox;
};

class rotate_y : public hitable
{
public:
    __device__ rotate_y(hitable *p, float angle);

    __device__ ~rotate_y();

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const override;

    __device__ aabb bounding_box() const override { return bbox; }

private:
    hitable *object;
    float sin_theta;
    float cos_theta;
    aabb bbox;
};

class hitable_list : public hitable
{
public:
    __device__ hitable_list() {}

    __device__ hitable_list(hitable **l, int n);

    __device__ ~hitable_list();

    __device__ int length();

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *local_rand_state) const override;

    __device__ aabb bounding_box() const override;

    __device__ float pdf_value(const vec3 &o, const vec3 &v, curandState *local_rand_state) const override;

    __device__ vec3 random(const vec3 &o, curandState *local_rand_state) const override;

private:
    hitable **list;
    int list_size = 0;
    aabb bbox;
};
