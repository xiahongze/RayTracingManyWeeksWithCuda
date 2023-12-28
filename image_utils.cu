#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"
#include "image_utils.h"

void writeJPGImage(const char *filename, int width, int height, vec3 *fb)
{
    uint8_t *imageData = new uint8_t[width * height * 3];
    size_t index = 0;
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            size_t pixel_index = j * width + i;
            imageData[index++] = int(255.99 * fmin(fb[pixel_index].r(), 1.0));
            imageData[index++] = int(255.99 * fmin(fb[pixel_index].g(), 1.0));
            imageData[index++] = int(255.99 * fmin(fb[pixel_index].b(), 1.0));
        }
    }

    stbi_write_jpg(filename, width, height, 3, imageData, 95);
    delete[] imageData;
}
