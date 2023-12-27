#include <iostream>
#include "cmd_parser.h"

CommandLineOptions parse_command_line(int argc, char **argv)
{
    CommandLineOptions options;

    cxxopts::Options cmd_options("Ray Tracer", "A simple ray tracer");
    cmd_options.add_options()                                                                                                        //
        ("o,output", "Output file (JPEG)", cxxopts::value<std::string>(options.output_file)->default_value(options.output_file))     //
        ("w,width", "Image width", cxxopts::value<int>(options.image_width)->default_value(options.image_width))                     //
        ("h,height", "Image height", cxxopts::value<int>(options.image_height)->default_value(options.image_height))                 //
        ("s,samples", "Samples per pixel", cxxopts::value<int>(options.samples_per_pixel)->default_value(options.samples_per_pixel)) //
        ("tx", "Threads in x direction", cxxopts::value<int>(options.tx)->default_value(options.tx))                                 //
        ("ty", "Threads in y direction", cxxopts::value<int>(options.ty)->default_value(options.ty))                                 //
        ("help", "Print help");

    try
    {
        auto result = cmd_options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << cmd_options.help() << std::endl;
            exit(0);
        }
    }
    catch (const cxxopts::exceptions::exception &e)
    {
        std::cout << "Error parsing options: " << e.what() << std::endl;
        exit(1);
    }

    return options;
}