#pragma once

#include "vec3.h"

void writeJPGImage(const char *filename, int width, int height, vec3 *fb);

unsigned char *readImage(const char *filename, int &width, int &height, int &channels);
