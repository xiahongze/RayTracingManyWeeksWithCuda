#pragma once

#include "external/cxxopts.hpp"

struct CommandLineOptions
{
    // general inputs
    std::string output_file;
    int image_width;
    int image_height;
    int samples_per_pixel;
    int tx;
    int ty;
    int choice;

    // scene inputs
    // week 1 world
    bool bounce;
    float bounce_pct;
    bool checkered;
};

CommandLineOptions parse_command_line(int argc, char **argv);
