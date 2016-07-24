//
// OpenCL NBody example
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

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#undef main

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>


#if !defined(_WIN32) && !defined(__APPLE__)
  #include <GL/glx.h>
#endif

#ifdef __APPLE__
  #include <OpenGL/OpenGL.h>
#endif



#include "util.hpp"
#include "err_code.h"
#include "device_picker.hpp"

#ifndef M_PI
  #define M_PI 3.14159265358979323846f
#endif

#if !defined(CL_VERSION_1_2)
  #define ImageGL Image2DGL
#endif

int      handleSDLEvents();
void     initGraphics();
void     parseArguments(int argc, char *argv[]);
void     releaseGraphics();
void     runReference(const std::vector<float>& initialPositions,
                      const std::vector<float>& initialVelocities,
                            std::vector<float>& finalPositions);

// Simulation parameters, with default values.
cl_uint  deviceIndex   =      0;
cl_uint  numBodies     =   4096;
cl_float delta         =      0.1f;
cl_float softening     =     10.f;
cl_uint  iterations    =     16;
float    sphereRadius  =    128;
float    tolerance     =      0.01f;
unsigned wgsize        =     64;
bool     useLocal      =     false;
unsigned init2D        =      0;
cl_uint  windowWidth   =    640;
cl_uint  windowHeight  =    480;

// SDL/GL objects
SDL_Window    *window;
SDL_GLContext  contextGL;
GLuint         textureGL;

int main(int argc, char *argv[])
{
  try
  {
    util::Timer timer;
    uint64_t startTime, endTime;

    parseArguments(argc, argv);

    initGraphics();

    // Initialize host data
    std::vector<float> h_initialPositions(4*numBodies);
    std::vector<float> h_initialVelocities(4*numBodies, 0);
    for (unsigned i = 0; i < numBodies; i++)
    {
      if (init2D)
      {
        // Generate a random point on the edge of a circle
        float angle = 2.f * (float)M_PI * (rand()/(float)RAND_MAX);
        h_initialPositions[i*4 + 0] = sphereRadius * cos(angle);
        h_initialPositions[i*4 + 1] = sphereRadius * sin(angle);
        h_initialPositions[i*4 + 2] = 0;
        h_initialPositions[i*4 + 3] = 1;
      }
      else
      {
        // Generate a random point on the surface of a sphere
        float longitude             = 2.f * (float)M_PI * (rand()/(float)RAND_MAX);
        float latitude              = acos((2.f * (rand()/(float)RAND_MAX)) - 1);
        h_initialPositions[i*4 + 0] = sphereRadius*sin(latitude)*cos(longitude);
        h_initialPositions[i*4 + 1] = sphereRadius*sin(latitude)*sin(longitude);
        h_initialPositions[i*4 + 2] = sphereRadius*cos(latitude);
        h_initialPositions[i*4 + 3] = 1;
      }
    }

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
    std::cout << std::endl << "Using OpenCL device: " << name << std::endl;

    cl_platform_id platform;
    device.getInfo(CL_DEVICE_PLATFORM, &platform);


#if defined(_WIN32)

    // Windows
    cl_context_properties properties[] = {
      CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
      CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
      0
    };

#elif defined(__APPLE__)

    // OS X
    CGLContextObj     kCGLContext     = CGLGetCurrentContext();
    CGLShareGroupObj  kCGLShareGroup  = CGLGetShareGroup(kCGLContext);

    cl_context_properties properties[] = {
      CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
      (cl_context_properties) kCGLShareGroup,
      0
    };

#else

    // Linux
    cl_context_properties properties[] = {
      CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
      CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
      0
    };

#endif

    cl::Context context(device, properties);
    cl::CommandQueue queue(context);

    cl::Program program(context, util::loadProgram("kernel.cl"));

    std::stringstream options;
    options.setf(std::ios::fixed, std::ios::floatfield);
    options << " -cl-fast-relaxed-math";
    options << " -cl-single-precision-constant";
    options << " -Dsoftening=" << softening << "f";
    options << " -Ddelta=" << delta << "f";
    options << " -DWGSIZE=" << wgsize;
    if (useLocal)
      options << " -DUSE_LOCAL";
    program.build(options.str().c_str());

    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_uint>
      nbodyKernel(program, "nbody");
    cl::KernelFunctor<cl::ImageGL>
      fillKernel(program, "fillTexture");
    cl::KernelFunctor<cl::Buffer, cl::ImageGL, cl_uint, cl_uint>
      drawKernel(program, "drawPositions");

    // Initialize device buffers
    cl::Buffer d_positions0, d_positions1, d_velocities;

    d_positions0 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              4*numBodies*sizeof(float));

    d_positions1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              4*numBodies*sizeof(float));

    d_velocities = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              4*numBodies*sizeof(float));

    cl::copy(queue, h_initialPositions.begin(), h_initialPositions.end(),
             d_positions0);
    cl::copy(queue, h_initialVelocities.begin(), h_initialVelocities.end(),
             d_velocities);

    cl::Buffer d_positionsIn  = d_positions0;
    cl::Buffer d_positionsOut = d_positions1;

    // Create CL image from GL texture
    cl::ImageGL d_texture(context, CL_MEM_WRITE_ONLY,
                          GL_TEXTURE_2D, 0, textureGL);
    std::vector<cl::Memory> clglObjects;
    clglObjects.push_back(d_texture);

    std::cout << "OpenCL initialization complete." << std::endl << std::endl;


    // Run simulation
    std::cout << "Running simulation..." << std::endl;
    startTime = timer.getTimeMicroseconds();
    cl::NDRange global(numBodies);
    cl::NDRange local(wgsize);
    cl::NDRange textureSize(windowWidth, windowHeight);
    size_t i;
    for (i = 0; ; i++)
    {
      nbodyKernel(cl::EnqueueArgs(queue, global, local),
                  d_positionsIn, d_positionsOut, d_velocities,
                  numBodies);

      // ***********************
      // Acquire texture from GL
      // ***********************
      glFlush();
      queue.enqueueAcquireGLObjects(&clglObjects);

      // Fill texture with a blank color
      fillKernel(cl::EnqueueArgs(queue, textureSize), d_texture);

      // Draw bodies
      drawKernel(cl::EnqueueArgs(queue, global, local),
                 d_positionsOut, d_texture, windowWidth, windowHeight);

      // **************************
      // Release texture back to GL
      // **************************
      queue.enqueueReleaseGLObjects(&clglObjects);
      queue.finish();

      // Render the texture as a quad filling the window
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, textureGL);
      glBegin(GL_QUADS);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
      glEnd();
      glDisable(GL_TEXTURE_2D);

      // Update window
      SDL_GL_SwapWindow(window);

      // Check for user input
      if (handleSDLEvents())
      {
        break;
      }

      // Swap position buffers
      cl::Buffer temp = d_positionsIn;
      d_positionsIn   = d_positionsOut;
      d_positionsOut  = temp;
    }

    endTime = timer.getTimeMicroseconds();
    std::cout << "OpenCL took " << ((endTime-startTime)*1e-3) << "ms"
              << std::endl << std::endl;

    std::cout << "Average FPS was " << (i / ((endTime-startTime)*1e-6))
              << std::endl << std::endl;

    releaseGraphics();
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

#if defined(_WIN32)
  system("pause");
#endif

  return 0;
}

int handleSDLEvents()
{
  SDL_Event e;
  while(SDL_PollEvent(&e))
  {
    // Quit on Q, Escape or window close
    if (SDL_KEYUP == e.type)
    {
      if (SDL_SCANCODE_Q == e.key.keysym.scancode)
        return 1;
      if (SDL_SCANCODE_ESCAPE == e.key.keysym.scancode)
        return 1;
    }
    if (SDL_QUIT == e.type)
      return 1;
  }

  return 0;
}

void initGraphics()
{
  if (SDL_Init(SDL_INIT_VIDEO) < 0)
  {
    fprintf(stderr, "Unable to init SDL: %s\n", SDL_GetError());
    exit(1);
  }

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

  window = SDL_CreateWindow(
    "nbody",
    SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
    windowWidth, windowHeight,
    SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN
  );

  if (NULL == window)
  {
    fprintf(stderr, "Unable to create SDL Window: %s\n", SDL_GetError());
    exit(1);
  }

  contextGL = SDL_GL_CreateContext(window);
  if (NULL == contextGL)
  {
    fprintf(stderr, "Unable to create OpenGL Context: %s\n", SDL_GetError());
    exit(1);
  }

  // Turn on Vsync
  SDL_GL_SetSwapInterval(1);

  // Create the texture
  glGenTextures(1, &textureGL);
  glBindTexture(GL_TEXTURE_2D, textureGL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowWidth, windowHeight, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
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
    else if (!strcmp(argv[i], "--numbodies") || !strcmp(argv[i], "-n"))
    {
      if (++i >= argc || !parseUInt(argv[i], &numBodies))
      {
        std::cout << "Invalid number of bodies" << std::endl;
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--delta") || !strcmp(argv[i], "-d"))
    {
      if (++i >= argc || !parseFloat(argv[i], &delta))
      {
        std::cout << "Invalid delta value" << std::endl;
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--softening") || !strcmp(argv[i], "-s"))
    {
      if (++i >= argc || !parseFloat(argv[i], &softening))
      {
        std::cout << "Invalid softening value" << std::endl;
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i"))
    {
      if (++i >= argc || !parseUInt(argv[i], &iterations))
      {
        std::cout << "Invalid number of iterations" << std::endl;
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--wgsize"))
    {
      if (++i >= argc || !parseUInt(argv[i], &wgsize))
      {
        std::cout << "Invalid work-group size" << std::endl;
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--local"))
    {
      useLocal = true;
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      std::cout << std::endl;
      std::cout << "Usage: ./nbody [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help               Print the message" << std::endl;
      std::cout << "      --list               List available devices" << std::endl;
      std::cout << "      --device     INDEX   Select device at INDEX" << std::endl;
      std::cout << "  -n  --numbodies  N       Run simulation with N bodies" << std::endl;
      std::cout << "  -d  --delta      DELTA   Time difference between iterations" << std::endl;
      std::cout << "  -s  --softening  SOFT    Force softening factor" << std::endl;
      std::cout << "  -i  --iterations ITRS    Run simulation for ITRS iterations" << std::endl;
      std::cout << "      --local              Enable use of local memory" << std::endl;
      std::cout << "      --wgsize     WGSIZE  Set work-group size to WGSIZE" << std::endl;
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

void releaseGraphics()
{
  SDL_GL_DeleteContext(contextGL);
  SDL_DestroyWindow(window);
  SDL_Quit();
}

void runReference(const std::vector<float>& initialPositions,
                  const std::vector<float>& initialVelocities,
                        std::vector<float>& finalPositions)
{
  std::vector<float> positions0 = initialPositions;
  std::vector<float> positions1(4*numBodies);
  std::vector<float> velocities = initialVelocities;

  std::vector<float>& positionsIn = positions0;
  std::vector<float>& positionsOut = positions1;

  for (unsigned itr = 0; itr < iterations; itr++)
  {
    for (unsigned i = 0; i < numBodies; i++)
    {
      float ix = positionsIn[i*4 + 0];
      float iy = positionsIn[i*4 + 1];
      float iz = positionsIn[i*4 + 2];
      float iw = positionsIn[i*4 + 3];

      float fx = 0.f;
      float fy = 0.f;
      float fz = 0.f;

      for (unsigned j = 0; j < numBodies; j++)
      {
        float jx    = positionsIn[j*4 + 0];
        float jy    = positionsIn[j*4 + 1];
        float jz    = positionsIn[j*4 + 2];
        float jw    = positionsIn[j*4 + 3];

        // Compute distance between bodies
        float dx    = (jx-ix);
        float dy    = (jy-iy);
        float dz    = (jz-iz);
        float dist  = sqrt(dx*dx + dy*dy + dz*dz + softening*softening);

        // Compute interaction force
        float coeff = jw / (dist*dist*dist);
        fx         += coeff * dx;
        fy         += coeff * dy;
        fz         += coeff * dz;
      }

      // Update velocity
      float vx            = velocities[i*4 + 0] + fx * delta;
      float vy            = velocities[i*4 + 1] + fy * delta;
      float vz            = velocities[i*4 + 2] + fz * delta;
      velocities[i*4 + 0] = vx;
      velocities[i*4 + 1] = vy;
      velocities[i*4 + 2] = vz;

      // Update position
      positionsOut[i*4 + 0] = ix + vx * delta;
      positionsOut[i*4 + 1] = iy + vy * delta;
      positionsOut[i*4 + 2] = iz + vz * delta;
      positionsOut[i*4 + 3] = iw;
    }

    // Swap buffers
    std::vector<float>& temp  = positionsIn;
    positionsIn  = positionsOut;
    positionsOut = temp;
  }

  finalPositions = positionsIn;
}
