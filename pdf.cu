#include "pdf.h"

__device__ cosine_pdf::cosine_pdf(const vec3 &w)
{
    uvw.build_from_w(w);
}

__device__ float cosine_pdf::value(const vec3 &direction, curandState *local_rand_state) const
{
    auto cosine_theta = dot(unit_vector(direction), uvw.w());
    return cosine_theta <= 0 ? 0 : cosine_theta / M_PI;
}

__device__ vec3 cosine_pdf::generate(curandState *local_rand_state) const
{
    return uvw.local(vec3::random_cosine_direction(local_rand_state));
}

__device__ float sphere_pdf::value(const vec3 &direction, curandState *local_rand_state) const
{
    return 1 / (4 * M_PI);
}

__device__ vec3 sphere_pdf::generate(curandState *local_rand_state) const
{
    return vec3::random_unit_vector(local_rand_state);
}

__device__ hitable_pdf::hitable_pdf(hitable *_objects, const vec3 &_origin)
    : objects(_objects), origin(_origin)
{
}

__device__ float hitable_pdf::value(const vec3 &direction, curandState *local_rand_state) const
{
    return objects->pdf_value(origin, direction, local_rand_state);
}

__device__ vec3 hitable_pdf::generate(curandState *local_rand_state) const
{
    return objects->random(origin, local_rand_state);
}

__device__ mixture_pdf::mixture_pdf(pdf *p0, pdf *p1)
{
    p[0] = p0;
    p[1] = p1;
}

__device__ float mixture_pdf::value(const vec3 &direction, curandState *local_rand_state) const
{
    return 0.5 * p[0]->value(direction, local_rand_state) + 0.5 * p[1]->value(direction, local_rand_state);
}

__device__ vec3 mixture_pdf::generate(curandState *local_rand_state) const
{
    if (curand_uniform(local_rand_state) < 0.5)
        return p[0]->generate(local_rand_state);
    else
        return p[1]->generate(local_rand_state);
}