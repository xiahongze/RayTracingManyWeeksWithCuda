# RayTracingTheNextWeekCuda

## Some learnings

### Why keeping `hittable **list` instead of `hittable *list`?

The reason behind this is actually complicated. Jump into the posts below for detailed explanations.

- [How to implement device side CUDA virtual functions?](https://stackoverflow.com/questions/26812913/how-to-implement-device-side-cuda-virtual-functions)
- [Polymorphism and derived classes in CUDA / CUDA Thrust](https://stackoverflow.com/questions/22988244/polymorphism-and-derived-classes-in-cuda-cuda-thrust/23476510#23476510)

My simple layman explanation is that,

- `hittable` is an abstract class containing a virtual function `hit`.
- `hittable_list` is a derived class of `hittable` containing a list of `hittable` objects.
- To use virtual functions in CUDA, we need to have the object instantiated in device memory.
- We can't just use cudaMalloc directly on host to allocate memory directly for the class and instead, we need to
  - cudaMalloc to allocate memory of a pointer to the class on host
  - pass this pointer to a kernel
  - use `new` to allocate memory for the class in device (that is how a class is instantiated in device memory)
  - or `malloc` directly in device memory in a kernel

The reason for using `hittable` is that we want to keep this simple interface such that objects of different types can be stored in the same list.
