#include "perlin.h"
#include "utils.h"

__device__ perlin::perlin(curandState *local_rand_state)
{
    ranvec = new vec3[point_count];
    for (int i = 0; i < point_count; ++i)
    {
        ranvec[i] = (-0.5f + vec3::random_cuda(local_rand_state)) * 2;
    }

    perm_x = perlin_generate_perm(local_rand_state);
    perm_y = perlin_generate_perm(local_rand_state);
    perm_z = perlin_generate_perm(local_rand_state);
}

__device__ perlin::~perlin()
{
    delete[] ranvec;
    delete[] perm_x;
    delete[] perm_y;
    delete[] perm_z;
}

__device__ float perlin::noise(const vec3 &p) const
{
    auto u = p.x() - floor(p.x());
    auto v = p.y() - floor(p.y());
    auto w = p.z() - floor(p.z());
    auto i = static_cast<int>(floor(p.x()));
    auto j = static_cast<int>(floor(p.y()));
    auto k = static_cast<int>(floor(p.z()));
    vec3 c[2][2][2];

    for (int di = 0; di < 2; di++)
        for (int dj = 0; dj < 2; dj++)
            for (int dk = 0; dk < 2; dk++)
                c[di][dj][dk] = ranvec[perm_x[(i + di) & 255] ^
                                       perm_y[(j + dj) & 255] ^
                                       perm_z[(k + dk) & 255]];

    return perlin_interp(c, u, v, w);
}

__device__ float perlin::turb(const vec3 &p, int depth) const
{
    auto accum = 0.0;
    auto temp_p = p;
    auto weight = 1.0;

    for (int i = 0; i < depth; i++)
    {
        accum += weight * noise(temp_p);
        weight *= 0.5;
        temp_p *= 2;
    }

    return fabs(accum);
}

__device__ int *perlin::perlin_generate_perm(curandState *local_rand_state)
{
    auto p = new int[point_count];

    for (int i = 0; i < point_count; i++)
        p[i] = i;

    permute(p, point_count, local_rand_state);

    return p;
}

__device__ void perlin::permute(int *p, int n, curandState *local_rand_state)
{
    for (int i = n - 1; i > 0; i--)
    {
        int target = (int)round(i * curand_uniform(local_rand_state));
        int tmp = p[i];
        p[i] = p[target];
        p[target] = tmp;
    }
}

__device__ float perlin::perlin_interp(vec3 c[2][2][2], float u, float v, float w)
{
    auto uu = u * u * (3 - 2 * u);
    auto vv = v * v * (3 - 2 * v);
    auto ww = w * w * (3 - 2 * w);
    auto accum = 0.0;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
            {
                vec3 weight_v(u - i, v - j, w - k);
                accum += (i * uu + (1 - i) * (1 - uu)) *
                         (j * vv + (1 - j) * (1 - vv)) *
                         (k * ww + (1 - k) * (1 - ww)) * dot(c[i][j][k], weight_v);
            }

    return accum;
}