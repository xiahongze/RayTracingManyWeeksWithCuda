#pragma once

#include "vec3.h"

class onb
{
public:
  __device__ onb() {}

  __device__ vec3 operator[](int i) const { return axis[i]; }
  __device__ vec3 &operator[](int i) { return axis[i]; }

  __device__ vec3 u() const { return axis[0]; }
  __device__ vec3 v() const { return axis[1]; }
  __device__ vec3 w() const { return axis[2]; }

  __device__ vec3 local(float a, float b, float c) const
  {
    return a * u() + b * v() + c * w();
  }

  __device__ vec3 local(const vec3 &a) const
  {
    return a.x() * u() + a.y() * v() + a.z() * w();
  }

  __device__ void build_from_w(const vec3 &w)
  {
    vec3 unit_w = unit_vector(w);
    vec3 a = (fabs(unit_w.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
    vec3 v = unit_vector(cross(unit_w, a));
    vec3 u = cross(unit_w, v);
    axis[0] = u;
    axis[1] = v;
    axis[2] = unit_w;
  }

private:
  vec3 axis[3];
};
