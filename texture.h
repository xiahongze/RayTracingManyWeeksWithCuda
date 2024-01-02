#pragma once

// #include "perlin.h"
// #include "rtw_stb_image.h"
#include "vec3.h"

namespace rtapp
{
  class texture
  {
  public:
    __host__ __device__ virtual vec3 value(float u, float v, const vec3 &p) const = 0;
  };

  class solid_color : public texture
  {
  public:
    __host__ __device__ solid_color(vec3 c);
    __host__ __device__ solid_color(float red, float green, float blue);
    __host__ __device__ vec3 value(float u, float v, const vec3 &p) const override;

  private:
    vec3 color_value;
  };

  class checker_texture : public texture
  {
  public:
    __host__ __device__ checker_texture(float _scale, texture *_even, texture *_odd);
    __host__ __device__ checker_texture(float _scale, vec3 c1, vec3 c2);
    __host__ __device__ ~checker_texture();
    __host__ __device__ vec3 value(float u, float v, const vec3 &p) const override;

  private:
    float inv_scale;
    texture *even;
    texture *odd;
  };
}

// class noise_texture : public texture
// {
// public:
//   noise_texture() {}

//   noise_texture(float sc) : scale(sc) {}

//   vec3 value(float u, float v, const vec3 &p) const override
//   {
//     auto s = scale * p;
//     return color(1, 1, 1) * 0.5 * (1 + sin(s.z() + 10 * noise.turb(s)));
//   }

// private:
//   perlin noise;
//   float scale;
// };

// class image_texture : public texture
// {
// public:
//   image_texture(const char *filename) : image(filename) {}

//   vec3 value(float u, float v, const vec3 &p) const override
//   {
//     // If we have no texture data, then return solid cyan as a debugging aid.
//     if (image.height() <= 0)
//       return color(0, 1, 1);

//     // Clamp input texture coordinates to [0,1] x [1,0]
//     u = interval(0, 1).clamp(u);
//     v = 1.0 - interval(0, 1).clamp(v); // Flip V to image coordinates

//     auto i = static_cast<int>(u * image.width());
//     auto j = static_cast<int>(v * image.height());
//     auto pixel = image.pixel_data(i, j);

//     auto color_scale = 1.0 / 255.0;
//     return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
//   }

// private:
//   rtw_image image;
// };