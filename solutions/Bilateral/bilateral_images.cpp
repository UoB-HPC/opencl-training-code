//
// OpenCL host<->device transfer exercise
//

/*
 *
 * This code is released under the "attribution CC BY" creative commons license.
 * In other words, you can use it in any way you see fit, including commercially,
 * but please retain an attribution for the original authors:
 * the High Performance Computing Group at the University of Bristol.
 * Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
 *
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef USE_SDL
#include <SDL2/SDL.h>
#else
typedef struct
{
  int w, h;
  unsigned char *pixels;
} HostImage;
HostImage* createHostImage(int width, int height);
#endif

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <device_picker.hpp>
#include <util.hpp>

#undef main
#undef min
#undef max

void parseArguments(int argc, char *argv[]);
void runReference(uint8_t *input, uint8_t *output, int width, int height);

// Parameters, with default values.
unsigned deviceIndex   =      0;
unsigned iterations    =     32;
unsigned tolerance     =      1;
bool     verify        =   true;
cl_int   radius        =      2;
float    sigmaDomain   =      3.f;
float    sigmaRange    =      0.2f;
#ifndef USE_SDL
int      width         =  1920;
int      height        =  1080;
#endif
cl::NDRange wgsize     = cl::NullRange;
const char *inputFile  =  "1080p.bmp";

int main(int argc, char *argv[])
{
  try
  {
    parseArguments(argc, argv);

    // Get list of devices
    std::vector<cl::Device> devices;
    getDeviceList(devices);

    // Check device index in range
    if (deviceIndex >= devices.size())
    {
      std::cout << "Invalid device index (try '--list')" << std::endl;
      return 1;
    }

    cl::Device device = devices[deviceIndex];

    std::string name = getDeviceName(device);
    std::cout << std::endl << "Using OpenCL device: " << name << std::endl
              << std::endl;

    cl_bool supportsImages = device.getInfo<CL_DEVICE_IMAGE_SUPPORT>();
    if (!supportsImages)
    {
       std::cout << std::endl << "Device doesn't support images!" << std::endl
                 << std::endl;
       return 1;
    }

    cl::Context context(device);
    cl::CommandQueue queue(context);
    cl::Program program(context, util::loadProgram("bilateral_images.cl"));

    std::stringstream options;
    options.setf(std::ios::fixed);
    options << " -cl-fast-relaxed-math";
    options << " -cl-single-precision-constant";
    options << " -DRADIUS=" << radius;
    options << " -DSIGMA_DOMAIN=" << sigmaDomain;
    options << " -DSIGMA_RANGE=" << sigmaRange;
    program.build(options.str().c_str());

    cl::KernelFunctor<cl::Image2D, cl::Image2D>
      kernel(program, "bilateral");

    // Load input image
#ifdef USE_SDL
    SDL_Surface *image = SDL_LoadBMP(inputFile);
    if (!image)
    {
      std::cout << SDL_GetError() << std::endl;
      throw;
    }
#else
    HostImage *image = createHostImage(width, height);
    for (int i = 0; i < image->w*image->h*4; i++)
    {
      image->pixels[i] = rand() % 256;
    }
#endif
    std::cout << "Processing image of size " << image->w << "x" << image->h
              << std::endl << std::endl;

    cl::ImageFormat format(CL_RGBA, CL_UNORM_INT8);
    cl::Image2D input(context, CL_MEM_READ_ONLY, format, image->w, image->h);
    cl::Image2D output(context, CL_MEM_WRITE_ONLY, format, image->w, image->h);

    // Write image to device
    cl::array<cl::size_type, 3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::array<cl::size_type, 3> region;
    region[0] = image->w;
    region[1] = image->h;
    region[2] = 1;
    queue.enqueueWriteImage(input, CL_TRUE, origin, region,
                            0, 0, image->pixels);


    cl::NDRange global(image->w, image->h);

    // Apply filter
    std::cout << "Running OpenCL..." << std::endl;
    util::Timer timer;
    uint64_t startTime = timer.getTimeMicroseconds();
    for (unsigned i = 0; i < iterations; i++)
    {
      kernel(cl::EnqueueArgs(queue, global, wgsize),
             input, output);
    }
    queue.finish();
    uint64_t endTime = timer.getTimeMicroseconds();
    double total = ((endTime-startTime)*1e-3);
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "OpenCL took " << total << "ms"
              << " (" << (total/iterations) << "ms / frame)"
              << std::endl << std::endl;

#ifdef USE_SDL
    // Save result to file
    SDL_Surface *result = SDL_ConvertSurface(image,
                                             image->format, image->flags);
    SDL_LockSurface(result);
    queue.enqueueReadImage(output, CL_TRUE, origin, region,
                           0, 0, result->pixels);
    SDL_UnlockSurface(result);
    SDL_SaveBMP(result, "output.bmp");
#else
    HostImage *result = createHostImage(image->w, image->h);
    queue.enqueueReadImage(output, CL_TRUE, origin, region,
                           0, 0, result->pixels);
#endif

    if (verify)
    {
      // Run reference
      std::cout << "Running reference..." << std::endl;
      uint8_t *reference = new uint8_t[image->w*image->h*4];
#ifdef USE_SDL
      SDL_LockSurface(image);
#endif
      startTime = timer.getTimeMicroseconds();
      runReference((uint8_t*)image->pixels, reference, image->w, image->h);
      endTime = timer.getTimeMicroseconds();
      std::cout << "Reference took " << ((endTime-startTime)*1e-3) << "ms"
                << std::endl << std::endl;

      // Check results
      char cstr[] = {'x', 'y', 'z'};
      unsigned errors = 0;
      for (int y = 0; y < result->h; y++)
      {
        for (int x = 0; x < result->w; x++)
        {
          for (int c = 0; c < 3; c++)
          {
            uint8_t out = ((uint8_t*)result->pixels)[(x + y*result->w)*4 + c];
            uint8_t ref = reference[(x + y*result->w)*4 + c];
            unsigned diff = abs((int)ref-(int)out);
            if (diff > tolerance)
            {
              if (!errors)
              {
                std::cout << "Verification failed:" << std::endl;
              }

              // Only show the first 8 errors
              if (errors++ < 8)
              {
                std::cout << "(" << x << "," << y << ")." << cstr[c] << ": "
                          << (int)out << " vs " << (int)ref << std::endl;
              }
            }
          }
        }
      }
      if (errors)
      {
        std::cout << "Total errors: " << errors << std::endl;
      }
      else
      {
        std::cout << "Verification passed." << std::endl;
      }
#ifdef USE_SDL
      SDL_UnlockSurface(result);
#endif

      delete[] reference;
    }
  }
  catch (cl::BuildError error)
  {
    std::string log = error.getBuildLog()[0].second;
    std::cerr << std::endl << "Build failed:" << std::endl << log << std::endl;
  }
  catch (cl::Error err)
  {
    std::cout << "Exception:" << std::endl
              << "ERROR: "
              << err.what()
              << "("
              << err_code(err.err())
              << ")"
              << std::endl;
  }
  std::cout << std::endl;

#if defined(_WIN32)
  system("pause");
#endif

  return 0;
}

int parseFloat(const char *str, cl_float *output)
{
  char *next;
  *output = (cl_float)strtod(str, &next);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--list"))
    {
      // Get list of devices
      std::vector<cl::Device> devices;
      getDeviceList(devices);

      // Print device names
      if (devices.size() == 0)
      {
        std::cout << "No devices found." << std::endl;
      }
      else
      {
        std::cout << std::endl;
        std::cout << "Devices:" << std::endl;
        for (unsigned i = 0; i < devices.size(); i++)
        {
          std::cout << i << ": " << getDeviceName(devices[i]) << std::endl;
        }
        std::cout << std::endl;
      }
      exit(0);
    }
    else if (!strcmp(argv[i], "--device"))
    {
      if (++i >= argc || !parseUInt(argv[i], &deviceIndex))
      {
        std::cout << "Invalid device index" << std::endl;
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--image"))
    {
      if (++i >= argc)
      {
        std::cout << "Missing argument to --image" << std::endl;
        exit(1);
      }
      inputFile = argv[i];
    }
    else if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i"))
    {
      if (++i >= argc || !parseUInt(argv[i], &iterations))
      {
        std::cout << "Invalid number of iterations" << std::endl;
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--noverify"))
    {
      verify = false;
    }
    else if (!strcmp(argv[i], "--sd"))
    {
      if (++i >= argc || !parseFloat(argv[i], &sigmaDomain))
      {
        std::cout << "Invalid sigma domain" << std::endl;
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--radius"))
    {
      if (++i >= argc || !parseUInt(argv[i], (cl_uint*)&radius))
      {
        std::cout << "Invalid radius" << std::endl;
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--sr"))
    {
      if (++i >= argc || !parseFloat(argv[i], &sigmaRange))
      {
        std::cout << "Invalid sigma range" << std::endl;
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--wgsize"))
    {
      unsigned width, height;
      if (++i >= argc || !parseUInt(argv[i], &width))
      {
        std::cout << "Invalid work-group width" << std::endl;
        exit(1);
      }
      if (++i >= argc || !parseUInt(argv[i], &height))
      {
        std::cout << "Invalid work-group height" << std::endl;
        exit(1);
      }
      wgsize = cl::NDRange(width, height);
    }
#ifndef USE_SDL
    else if (!strcmp(argv[i], "--width"))
    {
      if (++i >= argc || !parseUInt(argv[i], (cl_uint*)&width))
      {
        std::cout << "Invalid width" << std::endl;
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--height"))
    {
      if (++i >= argc || !parseUInt(argv[i], (cl_uint*)&height))
      {
        std::cout << "Invalid height" << std::endl;
        exit(1);
      }
    }
#endif
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      std::cout << std::endl;
      std::cout << "Usage: ./bilateral [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help               Print the message" << std::endl;
      std::cout << "      --list               List available devices" << std::endl;
      std::cout << "      --device     INDEX   Select device at INDEX" << std::endl;
      std::cout << "      --image      FILE    Use FILE as input (must be 32-bit RGBA)" << std::endl;
      std::cout << "  -i  --iterations ITRS    Number of benchmark iterations" << std::endl;
      std::cout << "      --noverify           Skip verification" << std::endl;
      std::cout << "      --radius     RADIUS  Set filter radius" << std::endl;
      std::cout << "      --sd         D       Set sigma domain" << std::endl;
      std::cout << "      --sr         R       Set sigma range" << std::endl;
      std::cout << "      --wgsize     W H     Work-group width and height" << std::endl;
#ifndef USE_SDL
      std::cout << "      --width      W       Set image width" << std::endl;
      std::cout << "      --height     H       Set image height" << std::endl;
#endif
      std::cout << std::endl;
      exit(0);
    }
    else
    {
      std::cout << "Unrecognized argument '" << argv[i] << "' (try '--help')"
                << std::endl;
      exit(1);
    }
  }
}

#ifndef USE_SDL
HostImage* createHostImage(int width, int height)
{
  HostImage *result = (HostImage*)malloc(sizeof(HostImage));
  result->w = width;
  result->h = height;
  result->pixels = (unsigned char*)malloc(width*height*4);
  return result;
}
#endif

void runReference(uint8_t *input, uint8_t *output,
                  int width, int height)
{
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      float cr = input[(x + y*width)*4 + 0]/255.f;
      float cg = input[(x + y*width)*4 + 1]/255.f;
      float cb = input[(x + y*width)*4 + 2]/255.f;

      float coeff = 0.f;
      float sr = 0.f;
      float sg = 0.f;
      float sb = 0.f;

      for (int j = -radius; j <= radius; j++)
      {
        for (int i = -radius; i <= radius; i++)
        {
          int xi = std::min(std::max(x+i, 0), width-1);
          int yj = std::min(std::max(y+j, 0), height-1);

          float r = input[(xi + yj*width)*4 + 0]/255.f;
          float g = input[(xi + yj*width)*4 + 1]/255.f;
          float b = input[(xi + yj*width)*4 + 2]/255.f;

          float weight, norm;

          norm = sqrt((float)(i*i) + (float)(j*j)) * (1.f/sigmaDomain);
          weight = exp(-0.5f * (norm*norm));

          norm = sqrt(pow(r-cr,2) + pow(g-cg,2) + pow(b-cb,2)) * (1.f/sigmaRange);
          weight *= exp(-0.5f * (norm*norm));

          coeff += weight;
          sr += weight * r;
          sg += weight * g;
          sb += weight * b;
        }
      }
      output[(x + y*width)*4 + 0] = (uint8_t)(std::min(std::max(sr/coeff, 0.f), 1.f)*255.f);
      output[(x + y*width)*4 + 1] = (uint8_t)(std::min(std::max(sg/coeff, 0.f), 1.f)*255.f);
      output[(x + y*width)*4 + 2] = (uint8_t)(std::min(std::max(sb/coeff, 0.f), 1.f)*255.f);
      output[(x + y*width)*4 + 3] = input[(x + y*width)*4 + 3];
    }
  }
}
