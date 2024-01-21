#pragma once

#include "hitable.h"
#include "onb.h"

enum pdf_type
{
  COSINE,
  SPHERE,
  HITABLE,
  MIXTURE
};

class pdf
{
public:
  __device__ virtual float value(const vec3 &direction, curandState *local_rand_state) const = 0;
  __device__ virtual vec3 generate(curandState *local_rand_state) const = 0;
};

class cosine_pdf : public pdf
{
public:
  __device__ cosine_pdf() {}

  __device__ cosine_pdf(const vec3 &w);

  __device__ float value(const vec3 &direction, curandState *local_rand_state) const override;

  __device__ vec3 generate(curandState *local_rand_state) const override;

private:
  onb uvw;
};

class sphere_pdf : public pdf
{
public:
  __device__ sphere_pdf() {}

  __device__ float value(const vec3 &direction, curandState *local_rand_state) const override;

  __device__ vec3 generate(curandState *local_rand_state) const override;
};

class hitable_pdf : public pdf
{
public:
  __device__ hitable_pdf() {}

  __device__ hitable_pdf(hitable *_objects, const vec3 &_origin);

  __device__ float value(const vec3 &direction, curandState *local_rand_state) const override;

  __device__ vec3 generate(curandState *local_rand_state) const override;

private:
  hitable *objects;
  vec3 origin;
};

class mixture_pdf : public pdf
{
public:
  __device__ mixture_pdf() {}

  __device__ mixture_pdf(pdf *p0, pdf *p1);

  __device__ float value(const vec3 &direction, curandState *local_rand_state) const override;

  __device__ vec3 generate(curandState *local_rand_state) const override;

private:
  pdf *p[2];
};
