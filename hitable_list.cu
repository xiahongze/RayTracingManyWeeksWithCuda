#include "hitable_list.h"

__device__ hitable_list::hitable_list() : list(nullptr), list_size(0) {}

__device__ hitable_list::hitable_list(hitable **l, int n) : list(l), list_size(n) {}

__device__ hitable_list::~hitable_list()
{
    for (int i = 0; i < list_size; i++)
    {
        delete list[i]; // list[i] was allocated with new
    }
}

__device__ bool hitable_list::hit(const ray &r, const interval ray_t, hit_record &rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.max;
    for (int i = 0; i < list_size; i++)
    {
        if (list[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}
