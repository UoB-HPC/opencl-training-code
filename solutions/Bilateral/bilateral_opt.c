//
// OpenCL bilateral filter exercise
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

#include <math.h>
#include <stdint.h>
#include <stdio.h>

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

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <device_picker.h>
#include <util.h>

#undef main
#undef min
#undef max

void parseArguments(int argc, char *argv[]);
void runReference(uint8_t *input, uint8_t *output, int width, int height);

// Parameters, with default values.
cl_uint  deviceIndex   =      0;
unsigned iterations    =     32;
unsigned tolerance     =      1;
int      verify        =      1;
cl_int   radius        =      2;
float    sigmaDomain   =      3.f;
float    sigmaRange    =      0.2f;
#ifndef USE_SDL
int      width         =  1920;
int      height        =  1080;
#endif
size_t *wgsize        = NULL;
const char *inputFile  =  "1080p.bmp";

int main(int argc, char *argv[])
{
  cl_int err;

  parseArguments(argc, argv);

  // Get list of devices
  cl_device_id devices[MAX_DEVICES];
  unsigned numDevices = getDeviceList(devices);

  // Check device index in range
  if (deviceIndex >= numDevices)
  {
    printf("Invalid device index (try '--list')\n");
    return 1;
  }

  cl_device_id device = devices[deviceIndex];

  char name[MAX_INFO_STRING];
  getDeviceName(device, name);
  printf("\nUsing OpenCL device: %s\n\n", name);

  // Create a compute context
  cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  checkError(err, "creating context");

  // Create a command queue
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  checkError(err, "creating command queue");

  // Create the program from the source buffer
  char *source = loadProgram("bilateral_opt.cl");
  cl_program program =
    clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
  checkError(err, "creating program");

  // Build the program
  char options[1024];
  sprintf(options,
    " -cl-fast-relaxed-math"
    " -cl-single-precision-constant"
    " -DRADIUS=%d"
    " -DSIGMA_DOMAIN=%.5ff"
    " -DSIGMA_RANGE=%.5ff",
    radius, sigmaDomain, sigmaRange);
  err = clBuildProgram(program, 1, &device, options, NULL, NULL);
  checkError(err, "building program");

  // Create the kernel
  cl_kernel kernel = clCreateKernel(program, "bilateral", &err);
  checkError(err, "creating kernel");

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
  printf("Processing image of size %dx%d\n\n", image->w, image->h);

  // Create buffers
  cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                image->w*image->h*4, NULL, &err);
  checkError(err, "creating input buffer");
  cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 image->w*image->h*4, NULL, &err);
  checkError(err, "creating output buffer");

  // Write image to device
  err = clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, image->w*image->h*4,
                             image->pixels, 0, NULL, NULL);
  checkError(err, "writing input image to device");

  // Set up kernel arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  checkError(err, "setting argument 0");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
  checkError(err, "setting argument 1");

  // Apply filter
  printf("Running OpenCL...\n");
  size_t global[2] = {width, height};
  double startTime = getCurrentTimeNanoseconds();
  for (unsigned i = 0; i < iterations; i++)
  {
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, wgsize,
                                 0, NULL, NULL);
    checkError(err, "enqueuing kernel");
  }
  err = clFinish(queue);
  checkError(err, "waiting for kernels to complete");
  double endTime = getCurrentTimeNanoseconds();
  double total = ((endTime-startTime)*1e-6);
  printf("OpenCL took %.1f ms (%.1f ms / frame)\n\n",
         total, (total/iterations));

#ifdef USE_SDL
  // Save result to file
  SDL_Surface *result = SDL_ConvertSurface(image,
                                           image->format, image->flags);
  SDL_LockSurface(result);
  queue.enqueueReadBuffer(output, CL_TRUE, 0,
                          image->w*image->h*4, result->pixels);
  SDL_UnlockSurface(result);
  SDL_SaveBMP(result, "output.bmp");
#else
  HostImage *result = createHostImage(image->w, image->h);
  err = clEnqueueReadBuffer(queue, output, CL_TRUE, 0, image->w*image->h*4,
                             result->pixels, 0, NULL, NULL);
  checkError(err, "reading output image from device");
#endif

  if (verify)
  {
    // Run reference
    printf("Running reference...\n");
    uint8_t *reference = malloc(image->w*image->h*4);
#ifdef USE_SDL
    SDL_LockSurface(image);
#endif
    startTime = getCurrentTimeNanoseconds();
    runReference((uint8_t*)image->pixels, reference, image->w, image->h);
    endTime = getCurrentTimeNanoseconds();
    total = ((endTime-startTime)*1e-6);
    printf("Reference took %.1f ms\n\n", total);

    // Check results
    char cstr[] = {'r', 'g', 'b'};
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
              printf("Verification failed:\n");
            }

            // Only show the first 8 errors
            if (errors++ < 8)
            {
              printf("(%d,%d).%c: %d vs %d\n",
                     x, y, cstr[c], (int)out, (int)ref);
            }
          }
        }
      }
    }
    if (errors)
    {
      printf("Total errors: %d\n", errors);
    }
    else
    {
      printf("Verification passed.\n");
    }
#ifdef USE_SDL
    SDL_UnlockSurface(result);
#endif

    free(reference);
  }
  printf("\n");

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
      cl_device_id devices[MAX_DEVICES];
      unsigned numDevices = getDeviceList(devices);

      // Print device names
      if (numDevices == 0)
      {
        printf("No devices found.\n");
      }
      else
      {
        printf("\nDevices:\n");
        for (int i = 0; i < numDevices; i++)
        {
          char name[MAX_INFO_STRING];
          getDeviceName(devices[i], name);
          printf("%2d: %s\n", i, name);
        }
        printf("\n");
      }
      exit(0);
    }
    else if (!strcmp(argv[i], "--device"))
    {
      if (++i >= argc || !parseUInt(argv[i], &deviceIndex))
      {
        printf("Invalid device index\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--image"))
    {
      if (++i >= argc)
      {
        printf("Missing argument to --image\n");
        exit(1);
      }
      inputFile = argv[i];
    }
    else if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i"))
    {
      if (++i >= argc || !parseUInt(argv[i], &iterations))
      {
        printf("Invalid number of iterations\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--noverify"))
    {
      verify = 0;
    }
    else if (!strcmp(argv[i], "--sd"))
    {
      if (++i >= argc || !parseFloat(argv[i], &sigmaDomain))
      {
        printf("Invalid sigma domain\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--radius"))
    {
      if (++i >= argc || !parseUInt(argv[i], (cl_uint*)&radius))
      {
        printf("Invalid radius\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--sr"))
    {
      if (++i >= argc || !parseFloat(argv[i], &sigmaRange))
      {
        printf("Invalid sigma range\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--wgsize"))
    {
      unsigned width, height;
      if (++i >= argc || !parseUInt(argv[i], &width))
      {
        printf("Invalid work-group width\n");
        exit(1);
      }
      if (++i >= argc || !parseUInt(argv[i], &height))
      {
        printf("Invalid work-group height\n");
        exit(1);
      }
      wgsize = malloc(sizeof(size_t) * 2);
      wgsize[0] = width;
      wgsize[1] = height;
    }
#ifndef USE_SDL
    else if (!strcmp(argv[i], "--width"))
    {
      if (++i >= argc || !parseUInt(argv[i], (cl_uint*)&width))
      {
        printf("Invalid width\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--height"))
    {
      if (++i >= argc || !parseUInt(argv[i], (cl_uint*)&height))
      {
        printf("Invalid height\n");
        exit(1);
      }
    }
#endif
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./bilateral [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h  --help               Print the message\n");
      printf("      --list               List available devices\n");
      printf("      --device     INDEX   Select device at INDEX\n");
      printf("      --image      FILE    Use FILE as input (must be 32-bit RGBA)\n");
      printf("  -i  --iterations ITRS    Number of benchmark iterations\n");
      printf("      --noverify           Skip verification\n");
      printf("      --radius     RADIUS  Set filter radius\n");
      printf("      --sd         D       Set sigma domain\n");
      printf("      --sr         R       Set sigma range\n");
      printf("      --wgsize     W H     Work-group width and height\n");
#ifndef USE_SDL
      printf("      --width      W       Set image width\n");
      printf("      --height     H       Set image height\n");
#endif
      printf("\n");
      exit(0);
    }
    else
    {
      printf("Unrecognized argument '%s' (try '--help')\n", argv[i]);
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

float clampFloat(float x, float low, float high)
{
  if (x < low)
    return low;
  if (x > high)
    return high;
  return x;
}

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
          int xi = (x+i) < 0 ? 0 : (x + i) >= width ? width - 1 : (x + i);
          int yj = (y+j) < 0 ? 0 : (y + j) >= height ? height - 1 : (y + j);

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
      output[(x + y*width)*4 + 0] = (uint8_t)(clampFloat(sr/coeff, 0.f, 1.f)*255.f);
      output[(x + y*width)*4 + 1] = (uint8_t)(clampFloat(sg/coeff, 0.f, 1.f)*255.f);
      output[(x + y*width)*4 + 2] = (uint8_t)(clampFloat(sb/coeff, 0.f, 1.f)*255.f);
      output[(x + y*width)*4 + 3] = input[(x + y*width)*4 + 3];
    }
  }
}
