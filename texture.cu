#include "image_utils.h"
#include "texture.h"
#include "utils.h"

// solid_color Implementation
__device__ solid_color::solid_color(vec3 c) : color_value(c) {}

__device__ solid_color::solid_color(float red, float green, float blue)
    : solid_color(vec3(red, green, blue)) {}

__device__ vec3 solid_color::value(float u, float v, const vec3 &p) const
{
    return color_value;
}

// checker_texture Implementation
__device__ checker_texture::checker_texture(float _scale, solid_color *_even, solid_color *_odd)
    : inv_scale(1.0 / _scale), even(_even), odd(_odd) {}

__device__ checker_texture::checker_texture(float _scale, vec3 c1, vec3 c2)
    : inv_scale(1.0 / _scale),
      even(new solid_color(c1)),
      odd(new solid_color(c2)) {}

__device__ checker_texture::~checker_texture()
{
    delete even;
    delete odd;
}

__device__ vec3 checker_texture::value(float u, float v, const vec3 &p) const
{
    auto xInteger = static_cast<int>(std::floor(inv_scale * p.x()));
    auto yInteger = static_cast<int>(std::floor(inv_scale * p.y()));
    auto zInteger = static_cast<int>(std::floor(inv_scale * p.z()));

    bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

    return isEven ? even->value(u, v, p) : odd->value(u, v, p);
}

// image_texture Implementation

__host__ image_texture::image_texture(const char *filename)
{
    pixel_data = readImage(filename, width, height, channels);
    pixel_data_size = width * height * channels * sizeof(unsigned char);
}

__device__ image_texture::image_texture(unsigned char *data, int w, int h, int c)
{
    pixel_data = data;
    width = w;
    height = h;
    channels = c;
    pixel_data_size = width * height * channels * sizeof(unsigned char);
}

__host__ __device__ image_texture::~image_texture()
{
    if (pixel_data)
    {
        delete[] pixel_data;
        pixel_data = nullptr;
    }
}

__device__ vec3 image_texture::value(float u, float v, const vec3 &p) const
{
    // If we have no texture data, then return solid cyan as a debugging aid.
    if (pixel_data == nullptr)
    {
        return vec3(0, 1, 1);
    }

    // Clamp input texture coordinates to [0,1] x [1,0]
    u = interval(0, 1).clamp(u);
    v = 1.0 - interval(0, 1).clamp(v); // Flip V to image coordinates

    auto i = static_cast<int>(u * width);
    auto j = static_cast<int>(v * height);

    // Clamp integer mapping, since actual coordinates should be less than 1.0
    if (i >= width)
    {
        i = width - 1;
    }
    if (j >= height)
    {
        j = height - 1;
    }

    const auto color_scale = 1.0f / 255.0f;
    auto pixel = pixel_data + j * width * channels + i * channels;

    return vec3(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
}

__device__ vec3 noise_texture::value(float u, float v, const vec3 &p) const
{
    auto s = scale * p;
    return vec3(1, 1, 1) * 0.5 * (1 + sin(s.z() + 10 * noise.turb(s)));
}

__device__ app_texture::app_texture(solid_color *s)
{
    type = texture_type::SOLID_COLOR;
    solid = s;
}

__device__ app_texture::app_texture(checker_texture *c)
{
    type = texture_type::CHECKER_TEXTURE;
    checker = c;
}

__device__ app_texture::app_texture(image_texture *i)
{
    type = texture_type::IMAGE_TEXTURE;
    image = i;
}

__device__ app_texture::app_texture(noise_texture *n)
{
    type = texture_type::NOISE_TEXTURE;
    noise = n;
}

__device__ vec3 app_texture::value(float u, float v, const vec3 &p) const
{
    switch (type)
    {
    case texture_type::SOLID_COLOR:
        return solid->value(u, v, p);
    case texture_type::CHECKER_TEXTURE:
        return checker->value(u, v, p);
    case texture_type::IMAGE_TEXTURE:
        return image->value(u, v, p);
    case texture_type::NOISE_TEXTURE:
        return noise->value(u, v, p);
    default:
        return vec3(0, 0, 0);
    }
}

__device__ app_texture::~app_texture()
{
    switch (type)
    {
    case texture_type::SOLID_COLOR:
        delete solid;
        break;
    case texture_type::CHECKER_TEXTURE:
        delete checker;
        break;
    case texture_type::IMAGE_TEXTURE:
        delete image;
        break;
    case texture_type::NOISE_TEXTURE:
        delete noise;
        break;
    default:
        break;
    }
}