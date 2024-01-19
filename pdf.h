#pragma once

#include "hitable.h"
#include "onb.h"

class pdf
{
public:
  virtual ~pdf() {}

  virtual double value(const vec3 &direction) const = 0;
  virtual vec3 generate() const = 0;
};

class cosine_pdf : public pdf
{
public:
  cosine_pdf(const vec3 &w) { uvw.build_from_w(w); }

  double value(const vec3 &direction) const override
  {
    auto cosine_theta = dot(unit_vector(direction), uvw.w());
    return fmax(0, cosine_theta / pi);
  }

  vec3 generate() const override
  {
    return uvw.local(random_cosine_direction());
  }

private:
  onb uvw;
};

class sphere_pdf : public pdf
{
public:
  sphere_pdf() {}

  double value(const vec3 &direction) const override
  {
    return 1 / (4 * pi);
  }

  vec3 generate() const override
  {
    return vec3::random_unit_vector();
  }
};

class hittable_pdf : public pdf
{
public:
  hittable_pdf(const hittable &_objects, const vec3 &_origin)
      : objects(_objects), origin(_origin)
  {
  }

  double value(const vec3 &direction) const override
  {
    return objects.pdf_value(origin, direction);
  }

  vec3 generate() const override
  {
    return objects.random(origin);
  }

private:
  const hitable &objects;
  vec3 origin;
};

class mixture_pdf : public pdf
{
public:
  mixture_pdf(pdf *p0, pdf *p1)
  {
    p[0] = p0;
    p[1] = p1;
  }

  double value(const vec3 &direction) const override
  {
    return 0.5 * p[0]->value(direction) + 0.5 * p[1]->value(direction);
  }

  vec3 generate() const override
  {
    if (random_double() < 0.5)
      return p[0]->generate();
    else
      return p[1]->generate();
  }

private:
  pdf *p[2];
};
