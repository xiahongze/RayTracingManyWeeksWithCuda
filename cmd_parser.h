#include "external/cxxopts.hpp"

struct CommandLineOptions
{
    std::string output_file = "output.jpg";
    int image_width = 1200;
    int image_height = 800;
    int samples_per_pixel = 10;
    int tx = 6;
    int ty = 4;
};

CommandLineOptions parse_command_line(int argc, char **argv);
