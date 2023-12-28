#pragma once

#include "hitable.h"

class hitable_list : public hitable
{
public:
    __device__ hitable_list();
    __device__ hitable_list(hitable **l, int n);
    __device__ bool hit(const ray &r, const interval ray_t, hit_record &rec) const override;
    __device__ ~hitable_list();

    hitable **list;
    int list_size;
};
