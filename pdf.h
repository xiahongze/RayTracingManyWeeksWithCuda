#pragma once

#include "hitable.h"
#include "onb.h"

class pdf
{
public:
  // virtual ~pdf() {}

  __device__ virtual float value(const vec3 &direction, curandState *local_rand_state) const = 0;
  __device__ virtual vec3 generate(curandState *local_rand_state) const = 0;
};

class cosine_pdf : public pdf
{
public:
  __device__ cosine_pdf(const vec3 &w) { uvw.build_from_w(w); }

  __device__ float value(const vec3 &direction, curandState *local_rand_state) const override
  {
    auto cosine_theta = dot(unit_vector(direction), uvw.w());
    return cosine_theta <= 0 ? 0 : cosine_theta / M_PI;
  }

  __device__ vec3 generate(curandState *local_rand_state) const override
  {
    return uvw.local(vec3::random_cosine_direction(local_rand_state));
  }

private:
  onb uvw;
};

class sphere_pdf : public pdf
{
public:
  __device__ sphere_pdf() {}

  __device__ float value(const vec3 &direction, curandState *local_rand_state) const override
  {
    return 1 / (4 * M_PI);
  }

  __device__ vec3 generate(curandState *local_rand_state) const override
  {
    return vec3::random_unit_vector(local_rand_state);
  }
};

class hitable_pdf : public pdf
{
public:
  __device__ hitable_pdf(const hitable &_objects, const vec3 &_origin)
      : objects(_objects), origin(_origin)
  {
  }

  __device__ float value(const vec3 &direction, curandState *local_rand_state) const override
  {
    return objects.pdf_value(origin, direction, local_rand_state);
  }

  __device__ vec3 generate(curandState *local_rand_state) const override
  {
    return objects.random(origin, local_rand_state);
  }

private:
  const hitable &objects;
  vec3 origin;
};

class mixture_pdf : public pdf
{
public:
  __device__ mixture_pdf(pdf *p0, pdf *p1)
  {
    p[0] = p0;
    p[1] = p1;
  }

  __device__ float value(const vec3 &direction, curandState *local_rand_state) const override
  {
    return 0.5 * p[0]->value(direction, local_rand_state) + 0.5 * p[1]->value(direction, local_rand_state);
  }

  __device__ vec3 generate(curandState *local_rand_state) const override
  {
    if (curand_normal(local_rand_state) < 0.5)
      return p[0]->generate(local_rand_state);
    else
      return p[1]->generate(local_rand_state);
  }

private:
  pdf *p[2];
};
