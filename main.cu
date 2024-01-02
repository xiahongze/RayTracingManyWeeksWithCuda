#include "cmd_parser.h"
#include "image_utils.h"
#include "render.h"
#include "scenes.h"
#include "utils.h"
#include "vec3.h"
#include <float.h>
#include <iostream>
#include <time.h>

int main(int argc, char **argv)
{
    auto cmd_opts = parse_command_line(argc, argv);

    int num_pixels = cmd_opts.image_width * cmd_opts.image_height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // make our world of hitables & the camera
    hitable **d_list;
    int list_size, tree_size;

    // create two arrays of bvh_nodes on host and device
    bvh_node *h_bvh_nodes, *d_bvh_nodes;

    camera *d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera)));

    int scene = 0;

    switch (scene)
    {
    case 0:
        random_spheres(h_bvh_nodes, d_bvh_nodes, d_list, d_camera, list_size, tree_size,
                       cmd_opts.image_width, cmd_opts.image_height, cmd_opts.bounce, cmd_opts.bounce_pct, cmd_opts.checkered);
        break;
    default:
        exit(1);
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // copy bvh_nodes from device to host
    checkCudaErrors(cudaMemcpy(h_bvh_nodes, d_bvh_nodes, list_size * sizeof(bvh_node), cudaMemcpyDeviceToHost));
    // build bvh tree on host
    int tree_height = bvh_node::build_tree(h_bvh_nodes, list_size);
    // copy bvh_nodes from host to device
    checkCudaErrors(cudaMemcpy(d_bvh_nodes, h_bvh_nodes, tree_size * sizeof(bvh_node), cudaMemcpyHostToDevice));

    clock_t start, stop;
    start = clock();

    // Render our buffer
    dim3 blocks(cmd_opts.image_width / cmd_opts.tx + (cmd_opts.image_width % cmd_opts.tx ? 1 : 0),
                cmd_opts.image_height / cmd_opts.ty + (cmd_opts.image_height % cmd_opts.ty ? 1 : 0));
    dim3 threads(cmd_opts.tx, cmd_opts.ty);
    render<<<blocks, threads>>>(fb, cmd_opts.image_width, cmd_opts.image_height, cmd_opts.samples_per_pixel, d_camera, d_bvh_nodes);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::clog << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image, allocated with cudaMallocManaged can be directly accessed on host
    writeJPGImage(cmd_opts.output_file.c_str(), cmd_opts.image_width, cmd_opts.image_height, fb);

    // clean up
    free_objects<<<dim3(1), dim3(32)>>>(d_list, list_size);
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_bvh_nodes));
    checkCudaErrors(cudaFree(fb));
    delete[] h_bvh_nodes;

    cudaDeviceReset();
}
