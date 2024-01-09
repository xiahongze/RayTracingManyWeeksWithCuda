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

### Bounding Volume Hierarchy (BVH)

Bounding Volume Hierarchy (BVH) is a tree structure commonly used in computer graphics, particularly in the fields of collision detection and ray tracing. This data structure allows for efficient representation and querying of a spatial scene by encapsulating geometry (such as triangles in a mesh) within bounding volumes, typically boxes or spheres. These volumes are organized hierarchically, with each node in the tree containing a volume that encompasses its children, leading to a fast exclusion of large parts of the scene when testing for intersections.

The linearization of BVH for GPU computation is a crucial optimization technique. GPUs, being massively parallel processors, prefer data structures with regular, predictable access patterns. Linearizing a BVH involves flattening the tree structure into a linear array that can be efficiently traversed by the GPU. This transformation typically involves ordering the nodes of the tree in a way that reduces memory jumps during traversal, which is critical for maintaining high performance in GPU-based computations. This linearization enables faster traversal speeds, making it highly suitable for real-time applications like gaming and interactive simulations, where rapid rendering is essential.

Here in this project, an array of `bvh_data_node` containing object information is passed from GPU to CPU and in CPU, we construct a tree of `_bvh_node`. After that, we linearize the tree into an array of `bvh_node` and pass it back to GPU. In GPU, we limit the stack size for non-recursive traversal. Adjust `MAX_TREE_HEIGHT` `in`bvh.h` to change the stack size if needed.

### Dynamical memory allocation in CUDA

In CUDA device functions, we can use `malloc` or `new` to allocate memory dynamically. However, it is advised that we use them with caution because it might cause unexpected race conditions. If you need to do so, do it with one thread for max safety. To avoid these issues, it's generally recommended to pre-allocate memory where possible, and to use shared memory for inter-thread communication within blocks. This approach minimizes the need for dynamic memory allocation and reduces the risk of race conditions.

### Copy & Move Semantics

Move semantics involves destructing the object that is being moved from. The class destructor is called right after the move constructor. While copy semantics does not call the destructor of the object that is being copied from, the destructor will be called after the object is out of scope or deleted. Take this example,

```
void foo() {
  A a = A(x, y, z); // move constructor, the rvalue will be destroyed after this line

  A b = a; // copy constructor, a is still valid after this line

  A *c = new A(a); // copy constructor, a is still valid after this line
  ...

  return; // a&b is destroyed here, but c lives on until it is deleted
}
```

The same principle applies to code in CUDA. If you have object that has a pointer and you want to keep the data that is pointed to, you need to either,

- do not release the memory in your class destructor
- implement a copy constructor and copy the data to the new object
- implement a move constructor and move the data to the new object, then set the pointer in the old object to `nullptr`

### Virtual functions & Memory allocation in CUDA

Subclasses need to be created dynamically in device function. If allocated using `cudaMalloc` in host, the pointer is not aligned and later
it causes issues when accessing members of the class. To avoid this, we need to use `new` to allocate memory for the class in device.

If this is a problem, consider avoiding using an abstract class at all.

### Virtual functions & debugging issues

If you are compiling with `-G` flag, you might encounter the following error when using virtual functions.

```
CUDA error = 719 at main.cu:75 'cudaDeviceSynchronize()'
```

This gives none information about the error as the problem apparently works fine without `-G` flag. To debug this, you need `-dopt=on` to optimize the code as well as `-G` to insert debug symbols. This is not ideal as removing optimization generally makes code more debuggable. However, in this case, it does the opposite.
