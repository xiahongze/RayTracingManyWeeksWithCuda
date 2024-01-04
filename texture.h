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

  class image_texture : public texture
  {
  public:
    __host__ image_texture(const char *filename);

    __host__ __device__ image_texture(unsigned char *data, int w, int h, int c);

    __host__ __device__ ~image_texture();

    __host__ __device__ vec3 value(float u, float v, const vec3 &p) const override;

    int width;
    int height;
    int channels = 3;
    unsigned char *pixel_data;
  };
  __global__ void update_image_texture(image_texture *texture, unsigned char *pixel_data, int width, int height, int channels);
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
