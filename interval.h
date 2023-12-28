#pragma once

#include <float.h>

// adapted from https://github.com/RayTracing/raytracing.github.io

class interval
{
public:
    float min, max;

    __host__ __device__ interval() : min(+FLT_MAX), max(-FLT_MAX) {} // Default interval is empty

    __host__ __device__ interval(float _min, float _max) : min(_min), max(_max) {}

    __host__ __device__ inline float size() const
    {
        return max - min;
    }

    __host__ __device__ inline interval expand(float delta) const
    {
        auto padding = delta / 2;
        return interval(min - padding, max + padding);
    }

    __host__ __device__ inline bool contains(float x) const
    {
        return min <= x && x <= max;
    }

    __host__ __device__ inline bool surrounds(float x) const
    {
        return min < x && x < max;
    }

    __host__ __device__ inline float clamp(float x) const
    {
        if (x < min)
            return min;
        if (x > max)
            return max;
        return x;
    }

    static const interval empty, universe;
};
