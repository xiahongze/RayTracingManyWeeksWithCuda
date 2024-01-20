#include "material.h"

// schlick
__device__ float schlick(float cosine, float ref_idx)
{
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

// refract
__device__ vec3 refract(const vec3 &uv, const vec3 &n, float etai_over_etat)
{
    auto cos_theta = fmin(dot(-uv, n), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.squared_length())) * n;
    return r_out_perp + r_out_parallel;
}

// reflect
__device__ vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2.0f * dot(v, n) * n;
}

// lambertian
__device__ lambertian::lambertian(const vec3 &a)
{
    albedo = new rtapp::solid_color(a);
}

__device__ lambertian::lambertian(rtapp::texture *a)
{
    albedo = a;
}

__device__ lambertian::~lambertian()
{
    delete albedo;
}

__device__ bool lambertian::scatter(const ray &r_in, const hit_record &rec, scatter_record &srec, curandState *local_rand_state) const
{
    // srec.attenuation = albedo->value(rec.u, rec.v, rec.p);
    // srec.pdf_ptr = new cosine_pdf(rec.normal); // this is a cause of slowness
    // srec.skip_pdf = false;

    // return true;

    /**
     * The following code is much faster than the above code, although not the same
     */
    vec3 scatter_direction = rec.normal + vec3::random_in_unit_sphere(local_rand_state);
    srec.skip_pdf = true;
    // Catch degenerate scatter direction
    if (scatter_direction.near_zero())
        scatter_direction = rec.normal;

    srec.skip_pdf_ray = ray(rec.p, scatter_direction, r_in.get_time());
    srec.attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
}

__device__ float lambertian::scattering_pdf(const ray &r_in, const hit_record &rec, const ray &scattered) const
{
    auto cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
    return cos_theta < 0 ? 0 : cos_theta / M_PI;
}

// metal
__device__ metal::metal(const vec3 &a, float f) : albedo(a.clamp()), fuzz(f < 1 ? f : 1) {}

__device__ bool metal::scatter(const ray &r_in, const hit_record &rec, scatter_record &srec, curandState *local_rand_state) const
{
    srec.attenuation = albedo;
    srec.pdf_ptr = nullptr;
    srec.skip_pdf = true;
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    srec.skip_pdf_ray =
        ray(rec.p, reflected + fuzz * vec3::random_in_unit_sphere(local_rand_state), r_in.get_time());
    return true;
}

// dielectric
__device__ dielectric::dielectric(float ri) : ref_idx(ri) {}

__device__ bool dielectric::scatter(const ray &r_in, const hit_record &rec, scatter_record &srec, curandState *local_rand_state) const
{
    srec.attenuation = vec3(1.0, 1.0, 1.0);
    srec.pdf_ptr = nullptr;
    srec.skip_pdf = true;
    float refraction_ratio = rec.front_face ? (1.0 / ref_idx) : ref_idx;

    vec3 unit_direction = unit_vector(r_in.direction());
    float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    vec3 direction;

    if (cannot_refract || schlick(cos_theta, refraction_ratio) > curand_uniform(local_rand_state))
        direction = reflect(unit_direction, rec.normal);
    else
        direction = refract(unit_direction, rec.normal, refraction_ratio);

    srec.skip_pdf_ray = ray(rec.p, direction);
    return true;
}

__device__ diffuse_light::diffuse_light(vec3 c) : emit(new rtapp::solid_color(c)) {}

__device__ diffuse_light::~diffuse_light()
{
    delete emit;
}

__device__ bool diffuse_light::scatter(const ray &r_in, const hit_record &rec, scatter_record &srec, curandState *local_rand_state) const
{
    return false;
}

__device__ vec3 diffuse_light::emitted(const ray &r_in, const hit_record &rec, float u, float v, const vec3 &p) const
{
    if (!rec.front_face)
        return vec3(0, 0, 0);
    return emit->value(u, v, p);
}

__device__ float diffuse_light::scattering_pdf(const ray &r_in, const hit_record &rec, const ray &scattered) const
{
    return 1 / (4 * M_PI);
}

__device__ isotropic::~isotropic()
{
    delete albedo;
}

__device__ bool isotropic::scatter(const ray &r_in, const hit_record &rec, scatter_record &srec, curandState *local_rand_state) const
{
    srec.attenuation = albedo->value(rec.u, rec.v, rec.p);
    srec.pdf_ptr = new sphere_pdf();
    srec.skip_pdf = false;
    return true;
}
