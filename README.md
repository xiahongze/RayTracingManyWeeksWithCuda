# RayTracingTheNextWeekCuda

## Some learnings

### Why keeping `hittable **list` instead of `hittable *list`?

This is because we allocate memory for `hittable` objects on the device and we want to keep the reference in host memory, so that we can pass this pointer of pointer to another device function (`__global__` function), which will then be able to access the `hittable` objects on the device. If we use `hittable *list`, then in the device function, we are not able to create an object and pass its pointer to the `hittable *list` on the device, because the object will be release after the function returns and also because setting `list = some_object_pointer` does not give us the some pointer after the function returns. See the code below:

- A good example

```cpp
__global__ void create_hittable(hittable **list, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        list = new hittable*[n];
        for (int i = 0; i < n; i++) {
            list[i] = new sphere(vec3(0, 0, 0), 1);
        }
    }
}

int main() {
    hittable **list;
    create_hittable<<<1, 1>>>(list, 1);
    ...
    // list will contains pointers of pointers which point to valid memory on device
}
```

- A bad example

```cpp
__global__ void create_hittable(hittable *list, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        list = new hittable[n];
        for (int i = 0; i < n; i++) {
            list[i] = sphere(vec3(0, 0, 0), 1);
        }

        // within this scope, list is a pointer to valid memory on device
        // however, after this function returns, the list passed in will not be updated
        // and the memory allocated in this scope will be dangling too.
    }
}

int main() {
    hittable *list;
    create_hittable<<<1, 1>>>(list, 1);
    ...
    // list will be a pointer to invalid memory on device
}
```

In summary, if you need to update the pointer passed in, you need to pass in a pointer of pointer. This is the case when
we need to have a reference on host memory to the memory allocated on device.
