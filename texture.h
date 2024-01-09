#pragma once

#include "perlin.h"
#include "vec3.h"

class solid_color
{
public:
  __device__ solid_color(vec3 c);
  __device__ solid_color(float red, float green, float blue);
  __device__ vec3 value(float u, float v, const vec3 &p) const;

private:
  vec3 color_value;
};

class checker_texture
{
public:
  __device__ checker_texture(float _scale, solid_color *_even, solid_color *_odd);
  __device__ checker_texture(float _scale, vec3 c1, vec3 c2);
  __device__ ~checker_texture();
  __device__ vec3 value(float u, float v, const vec3 &p) const;

private:
  float inv_scale;
  solid_color *even;
  solid_color *odd;
};

class image_texture
{
public:
  __host__ image_texture(const char *filename);

  __device__ image_texture(unsigned char *data, int w, int h, int c);

  __host__ __device__ ~image_texture();

  __device__ vec3 value(float u, float v, const vec3 &p) const;

  int width;
  int height;
  int channels = 3;
  unsigned char *pixel_data;
  int pixel_data_size;
};

class noise_texture
{
public:
  __device__ noise_texture() {}

  __device__ noise_texture(float sc) : scale(sc) {}

  __device__ vec3 value(float u, float v, const vec3 &p) const;

private:
  perlin noise;
  float scale;
};

enum texture_type
{
  SOLID_COLOR,
  CHECKER_TEXTURE,
  IMAGE_TEXTURE,
  NOISE_TEXTURE
};

class app_texture
{
public:
  __device__ app_texture(solid_color *s);
  __device__ app_texture(checker_texture *c);
  __device__ app_texture(image_texture *i);
  __device__ app_texture(noise_texture *n);
  __device__ ~app_texture();

  __device__ vec3 value(float u, float v, const vec3 &p) const;

private:
  texture_type type;
  solid_color *solid;
  checker_texture *checker;
  image_texture *image;
  noise_texture *noise;
};
