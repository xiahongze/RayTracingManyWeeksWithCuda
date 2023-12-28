#pragma once

#include "aabb.h"
#include "hitable.h"

class hitable_list
{
public:
    __device__ hitable_list();
    __device__ hitable_list(hitable **l, int n);
    // same as hitable::hit but not a subclass of hitable
    __device__ bool hit(const ray &r, const interval ray_t, hit_record &rec) const;
    __device__ aabb bounding_box() const;
    __device__ void set_aabb();
    __device__ ~hitable_list();

private:
    hitable **list;
    int list_size;
    aabb bbox;
};
