#include "cmd_parser.h"
#include <iostream>

CommandLineOptions parse_command_line(int argc, char **argv)
{
    CommandLineOptions options;

    cxxopts::Options cmd_options("Ray Tracer", "A simple ray tracer");
    cmd_options.add_options()
        // general inputs
        ("o,output", "Output file (JPEG)", cxxopts::value<std::string>(options.output_file)->default_value("out.jpg")) //
        ("w,width", "Image width", cxxopts::value<int>(options.image_width)->default_value("1200"))                    //
        ("h,height", "Image height", cxxopts::value<int>(options.image_height)->default_value("800"))                  //
        ("s,samples", "Samples per pixel", cxxopts::value<int>(options.samples_per_pixel)->default_value("10"))        //
        ("tx", "Threads in x direction", cxxopts::value<int>(options.tx)->default_value("6"))                          //
        ("ty", "Threads in y direction", cxxopts::value<int>(options.ty)->default_value("4"))                          //
        // week 1 scene options
        ("wk1-bounce", "Enable bouncing spheres", cxxopts::value<bool>(options.bounce)->default_value("false"))                //
        ("wk1-bounce-pct", "Percentage of bouncing spheres", cxxopts::value<float>(options.bounce_pct)->default_value("0.33")) //
        ("help", "Print help");

    try
    {
        auto result = cmd_options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << cmd_options.help() << std::endl;
            exit(0);
        }

        std::cout << "Rendering a " << options.image_width << "x"
                  << options.image_height << " image with " << options.samples_per_pixel << " samples per pixel ";
        std::cout << "in " << options.tx << "x" << options.ty << " blocks.\n";
        std::cout << "Output file: " << options.output_file << "\n";
    }
    catch (const cxxopts::exceptions::exception &e)
    {
        std::cout << "Error parsing options: " << e.what() << std::endl;
        exit(1);
    }

    return options;
}
