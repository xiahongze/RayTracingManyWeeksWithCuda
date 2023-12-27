#pragma once

#include "vec3.h"

class ray
{
public:
    __device__ ray() {}
    __device__ ray(const vec3 &a, const vec3 &b) : A(a), B(b), time(0) {}
    __device__ ray(const vec3 &a, const vec3 &b, const float time) : A(a), B(b), time(time) {}

    __device__ inline vec3 origin() const { return A; }
    __device__ inline vec3 direction() const { return B; }
    __device__ inline vec3 point_at_parameter(float t) const { return A + t * B; }
    __device__ inline float get_time() const { return time; }

private:
    vec3 A;
    vec3 B;
    float time;
};
