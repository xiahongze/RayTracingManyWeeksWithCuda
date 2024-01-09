#include "hitable.h"
#include "material.h"
#include "texture.h"

class constant_medium : public hitable
{
public:
  __device__ constant_medium(hitable *b, float d, rtapp::texture *a);

  __device__ constant_medium(hitable *b, float d, vec3 c);

  __device__ ~constant_medium();

  __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec) const override;

  __device__ aabb bounding_box() const override { return boundary->bounding_box(); }

private:
  hitable *boundary;
  float neg_inv_density;
  material *phase_function;
};
