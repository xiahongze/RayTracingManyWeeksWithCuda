#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"
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
            auto pixel = fb[pixel_index].clamp();
            imageData[index++] = int(255.99 * pixel.r());
            imageData[index++] = int(255.99 * pixel.g());
            imageData[index++] = int(255.99 * pixel.b());
        }
    }

    stbi_write_jpg(filename, width, height, 3, imageData, 95);
    delete[] imageData;
}

unsigned char *readImage(const char *filename, int &width, int &height, int &channels)
{
    // int n;
    unsigned char *data = stbi_load(filename, &width, &height, &channels, 0);
    if (data == NULL)
    {
        std::cerr << "Error: could not load image " << filename << std::endl;
        exit(1);
    }
    return data;
}
