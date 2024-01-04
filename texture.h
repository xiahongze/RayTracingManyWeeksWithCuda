#pragma once

#include "perlin.h"
#include "vec3.h"

namespace rtapp
{
  class texture
  {
  public:
    __device__ virtual vec3 value(float u, float v, const vec3 &p) const = 0;
  };

  class solid_color : public texture
  {
  public:
    __device__ solid_color(vec3 c);
    __device__ solid_color(float red, float green, float blue);
    __device__ vec3 value(float u, float v, const vec3 &p) const override;

  private:
    vec3 color_value;
  };

  class checker_texture : public texture
  {
  public:
    __device__ checker_texture(float _scale, texture *_even, texture *_odd);
    __device__ checker_texture(float _scale, vec3 c1, vec3 c2);
    __device__ ~checker_texture();
    __device__ vec3 value(float u, float v, const vec3 &p) const override;

  private:
    float inv_scale;
    texture *even;
    texture *odd;
  };

  class image_texture : public texture
  {
  public:
    __host__ image_texture(const char *filename);

    __device__ image_texture(unsigned char *data, int w, int h, int c);

    __host__ __device__ ~image_texture();

    __device__ vec3 value(float u, float v, const vec3 &p) const override;

    int width;
    int height;
    int channels = 3;
    unsigned char *pixel_data;
    int pixel_data_size;
  };

  class noise_texture : public texture
  {
  public:
    __device__ noise_texture() {}

    __device__ noise_texture(float sc) : scale(sc) {}

    __device__ vec3 value(float u, float v, const vec3 &p) const override;

  private:
    perlin noise;
    float scale;
  };
}
