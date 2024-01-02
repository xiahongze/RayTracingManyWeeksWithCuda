#include "texture.h"

namespace rtapp
{
    // solid_color Implementation
    __host__ __device__ solid_color::solid_color(vec3 c) : color_value(c) {}

    __host__ __device__ solid_color::solid_color(float red, float green, float blue)
        : solid_color(vec3(red, green, blue)) {}

    __host__ __device__ vec3 solid_color::value(float u, float v, const vec3 &p) const
    {
        return color_value;
    }

    // checker_texture Implementation
    __host__ __device__ checker_texture::checker_texture(float _scale, texture *_even, texture *_odd)
        : inv_scale(1.0 / _scale), even(_even), odd(_odd) {}

    __host__ __device__ checker_texture::checker_texture(float _scale, vec3 c1, vec3 c2)
        : inv_scale(1.0 / _scale),
          even(new solid_color(c1)),
          odd(new solid_color(c2)) {}

    __host__ __device__ checker_texture::~checker_texture()
    {
        delete even;
        delete odd;
    }

    __host__ __device__ vec3 checker_texture::value(float u, float v, const vec3 &p) const
    {
        auto xInteger = static_cast<int>(std::floor(inv_scale * p.x()));
        auto yInteger = static_cast<int>(std::floor(inv_scale * p.y()));
        auto zInteger = static_cast<int>(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }
}
