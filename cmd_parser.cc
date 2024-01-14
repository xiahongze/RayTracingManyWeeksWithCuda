#include "cmd_parser.h"
#include <iostream>

CommandLineOptions parse_command_line(int argc, char **argv)
{
    CommandLineOptions options;

    cxxopts::Options cmd_options("Ray Tracer", "A simple ray tracer");

    const auto scene_option_desc = "choice of scenes:\n0. random spheres\n1. globe\n2. two perlin spheres\n3. quads\n4. simple light\n5. cornell box\n6. final scene week2\n";
    cmd_options.add_options()
        // general inputs
        ("o,output", "Output file (JPEG)", cxxopts::value<std::string>(options.output_file)->default_value("out.jpg")) //
        ("w,width", "Image width", cxxopts::value<int>(options.image_width)->default_value("1200"))                    //
        ("h,height", "Image height", cxxopts::value<int>(options.image_height)->default_value("800"))                  //
        ("s,samples", "Samples per pixel", cxxopts::value<int>(options.samples_per_pixel)->default_value("10"))        //
        ("tx", "Threads in x direction", cxxopts::value<int>(options.tx)->default_value("6"))                          //
        ("ty", "Threads in y direction", cxxopts::value<int>(options.ty)->default_value("4"))                          //
        ("c,choice", scene_option_desc, cxxopts::value<int>(options.choice)->default_value("0"))                       //
        ("seed", "Random seed", cxxopts::value<int>(options.seed)->default_value("1984"))                              //
        ("max-depth", "max number of bounce", cxxopts::value<int>(options.max_depth)->default_value("50"))             //
        // week 1 scene options
        ("wk1-bounce", "Enable bouncing spheres", cxxopts::value<bool>(options.bounce)->default_value("false"))                                           //
        ("wk1-bounce-pct", "Percentage of bouncing spheres", cxxopts::value<float>(options.bounce_pct)->default_value("0.33"))                            //
        ("wk1-checker", "Use checker board ground", cxxopts::value<bool>(options.checkered)->default_value("false"))                                      //
        ("wk2-rotate-translate", "Whether to rotate and translate the boxes", cxxopts::value<bool>(options.cornell_box_rt_trans)->default_value("false")) //
        ("wk2-smoke", "Add smoke to cornell box", cxxopts::value<bool>(options.cornell_box_smoke)->default_value("false"))                                //
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
