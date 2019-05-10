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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#define GL_SILENCE_DEPRECATION
#endif
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#undef main

#if defined(__APPLE__)
  #define CL_SILENCE_DEPRECATION
  #include <OpenCL/OpenCL.h>
  #include <OpenGL/OpenGL.h>
#else
  #define CL_TARGET_OPENCL_VERSION 120
  #include <CL/cl.h>
  #include <CL/cl_ext.h>
  #include <CL/cl_gl.h>
  #include <GL/gl.h>
#if !defined(_WIN32)
  #include <GL/glx.h>
#endif
#endif

#include <device_picker.h>
#include <util.h>

#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

int      handleSDLEvents();
void     initGraphics();
void     parseArguments(int argc, char *argv[]);
void     releaseGraphics();
void     runReference(const float *initialPositions,
                      const float *initialVelocities,
                      float *finalPositions);

// Simulation parameters, with default values.
cl_uint  deviceIndex   =      0;
cl_uint  numBodies     =   4096;
cl_float delta         =      0.1f;
cl_float softening     =     10.f;
cl_uint  iterations    =     16;
float    sphereRadius  =    128;
float    tolerance     =      0.01f;
unsigned wgsize        =     64;
int      useLocal      =      0;
unsigned init2D        =      0;
cl_uint  windowWidth   =    640;
cl_uint  windowHeight  =    480;

// SDL/GL objects
SDL_Window    *window;
SDL_GLContext  contextGL;
GLuint         textureGL;

int main(int argc, char *argv[])
{
  cl_int           err;
  cl_device_id     device;
  cl_context       context;
  cl_command_queue queue;
  cl_program       program;
  cl_kernel        nbodyKernel, fillKernel, drawKernel;
  double           start, end;

  parseArguments(argc, argv);

  initGraphics();

  // Initialize host data
  size_t dataSize            = numBodies*sizeof(cl_float4);
  float *h_initialPositions  = malloc(dataSize);
  float *h_initialVelocities = malloc(dataSize);
  float *h_positions         = NULL;
  for (int i = 0; i < numBodies; i++)
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
  memset(h_initialVelocities, 0, dataSize);

  // Get list of devices
  cl_device_id devices[MAX_DEVICES];
  unsigned numDevices = getDeviceList(devices);

  // Check device index in range
  if (deviceIndex >= numDevices)
  {
    printf("Invalid device index (try '--list')\n");
    return 1;
  }

  device = devices[deviceIndex];

  char name[MAX_INFO_STRING];
  getDeviceName(device, name);
  printf("\nUsing OpenCL device: %s\n", name);

  cl_platform_id platform;
  err = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
                        sizeof(cl_platform_id), &platform, NULL);
  checkError(err, "getting platform");


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

  context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
  checkError(err, "creating cl/gl shared context");

  queue = clCreateCommandQueue(context, device, 0, &err);
  checkError(err, "creating command queue");

  char *source = loadProgram("kernel.cl");
  program = clCreateProgramWithSource(context, 1, (const char **)&source,
                                      NULL, &err);
  checkError(err, "creating program");

  char options[256];
  sprintf(options,
          "-cl-fast-relaxed-math -cl-single-precision-constant "
          "-Dsoftening=%ff -Ddelta=%ff -DWGSIZE=%d %s",
          softening, delta, wgsize, useLocal ? "-DUSE_LOCAL" : "");
  err = clBuildProgram(program, 1, &device, options, NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t sz;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          0, NULL, &sz);
    char *buildLog = malloc(++sz);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sz, buildLog, NULL);
    fprintf(stderr, "%s\n", buildLog);
    free(buildLog);
  }
  checkError(err, "building program");

  nbodyKernel = clCreateKernel(program, "nbody", &err);
  checkError(err, "creating nbody kernel");

  fillKernel = clCreateKernel(program, "fillTexture", &err);
  checkError(err, "creating fill kernel");

  drawKernel = clCreateKernel(program, "drawPositions", &err);
  checkError(err, "creating draw positions kernel");

  // Initialize device buffers
  cl_mem d_positions0, d_positions1, d_velocities;

  d_positions0 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                dataSize, NULL, &err);
  checkError(err, "creating d_positions0 buffer");

  d_positions1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                dataSize, NULL, &err);
  checkError(err, "creating d_positions1 buffer");

  d_velocities = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                dataSize, NULL, &err);
  checkError(err, "creating d_velocities buffer");

  err = clEnqueueWriteBuffer(queue, d_positions0, CL_TRUE,
                             0, dataSize, h_initialPositions, 0, NULL, NULL);
  checkError(err, "writing d_positions data");
  err = clEnqueueWriteBuffer(queue, d_velocities, CL_TRUE,
                             0, dataSize, h_initialVelocities, 0, NULL, NULL);
  checkError(err, "writing d_velocities data");

  err  = clSetKernelArg(nbodyKernel, 2, sizeof(cl_mem), &d_velocities);
  err |= clSetKernelArg(nbodyKernel, 3, sizeof(cl_uint), &numBodies);
  checkError(err, "setting nbody kernel args");

  cl_mem d_positionsIn  = d_positions0;
  cl_mem d_positionsOut = d_positions1;

  // Create CL image from GL texture
  cl_mem d_texture = clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY,
                                           GL_TEXTURE_2D, 0, textureGL, &err);
  checkError(err, "creating CL image from GL texture");

  err  = clSetKernelArg(fillKernel, 0, sizeof(cl_mem), &d_texture);
  checkError(err, "setting fill kernel args");

  err  = clSetKernelArg(drawKernel, 1, sizeof(cl_mem), &d_texture);
  err |= clSetKernelArg(drawKernel, 2, sizeof(cl_uint), &windowWidth);
  err |= clSetKernelArg(drawKernel, 3, sizeof(cl_uint), &windowHeight);
  checkError(err, "setting draw kernel args");

  printf("OpenCL initialization complete.\n\n");


  // Run simulation
  printf("Running simulation...\n");
  start = getCurrentTimeNanoseconds();
  size_t global[1]      = {numBodies};
  size_t local[1]       = {wgsize};
  size_t textureSize[2] = {windowWidth,windowHeight};
  size_t i;
  for (i = 0; ; i++)
  {
    // Enqueue nbody kernel
    err  = clSetKernelArg(nbodyKernel, 0, sizeof(cl_mem), &d_positionsIn);
    err |= clSetKernelArg(nbodyKernel, 1, sizeof(cl_mem), &d_positionsOut);
    err |= clEnqueueNDRangeKernel(queue, nbodyKernel,
                                  1, NULL, global, local, 0, NULL, NULL);
    checkError(err, "enqueuing nbody kernel");

    // Flush GL queue and acquire texture
    glFlush();
    err = clEnqueueAcquireGLObjects(queue, 1, &d_texture, 0, NULL, NULL);
    checkError(err, "acquiring GL objects");

    // Fill texture with a blank color
    err = clEnqueueNDRangeKernel(queue, fillKernel,
                                 2, NULL, textureSize, NULL, 0, NULL, NULL);
    checkError(err, "enqueueing fill kernel");

    // Draw bodies
    err = clSetKernelArg(drawKernel, 0, sizeof(cl_mem), &d_positionsOut);
    err = clEnqueueNDRangeKernel(queue, drawKernel,
                                 1, NULL, global, local, 0, NULL, NULL);
    checkError(err, "enqueuing draw kernel");

    // Release texture
    err = clEnqueueReleaseGLObjects(queue, 1, &d_texture, 0, NULL, NULL);
    checkError(err, "releasing GL objects");

    // Finish CL queue
    err = clFinish(queue);
    checkError(err, "finishing CL queue");

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
    cl_mem temp    = d_positionsIn;
    d_positionsIn  = d_positionsOut;
    d_positionsOut = temp;
  }

  // Read final positions
  h_positions = clEnqueueMapBuffer(queue, d_positionsIn, CL_FALSE, CL_MAP_READ,
                                   0, dataSize, 0, NULL, NULL, &err);
  checkError(err, "mapping positions buffer");

  err = clEnqueueReadBuffer(queue, d_positionsIn, CL_FALSE,
                            0, dataSize, h_positions, 0, NULL, NULL);
  checkError(err, "reading final positions");

  err = clFinish(queue);
  checkError(err, "running kernel");

  end = getCurrentTimeNanoseconds();
  printf("OpenCL took %.2lfms\n\n", (end-start)*1e-6);

  printf("Average FPS was %.1f\n\n", i / ((end-start)*1e-9));

  // Unmap memory objects
  err = clEnqueueUnmapMemObject(queue, d_positionsIn, h_positions, 0, NULL, NULL);
  checkError(err, "unmapping positions buffer");

  free(h_initialPositions);
  free(h_initialVelocities);
  clReleaseMemObject(d_positions0);
  clReleaseMemObject(d_positions1);
  clReleaseMemObject(d_velocities);
  clReleaseKernel(nbodyKernel);
  clReleaseKernel(drawKernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  releaseGraphics();

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
      cl_device_id devices[MAX_DEVICES];
      unsigned numDevices = getDeviceList(devices);

      // Print device names
      if (numDevices == 0)
      {
        printf("No devices found.\n");
      }
      else
      {
        printf("\n");
        printf("Devices:\n");
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
    else if (!strcmp(argv[i], "--numbodies") || !strcmp(argv[i], "-n"))
    {
      if (++i >= argc || !parseUInt(argv[i], &numBodies))
      {
        printf("Invalid number of bodies\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--delta") || !strcmp(argv[i], "-d"))
    {
      if (++i >= argc || !parseFloat(argv[i], &delta))
      {
        printf("Invalid delta value\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--softening") || !strcmp(argv[i], "-s"))
    {
      if (++i >= argc || !parseFloat(argv[i], &softening))
      {
        printf("Invalid softening value\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i"))
    {
      if (++i >= argc || !parseUInt(argv[i], &iterations))
      {
        printf("Invalid number of iterations\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--wgsize"))
    {
      if (++i >= argc || !parseUInt(argv[i], &wgsize))
      {
        printf("Invalid work-group size\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--width"))
    {
      if (++i >= argc || !parseUInt(argv[i], &windowWidth))
      {
        printf("Invalid window width\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--height"))
    {
      if (++i >= argc || !parseUInt(argv[i], &windowHeight))
      {
        printf("Invalid window height\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--2D"))
    {
      init2D = 1;
    }
    else if (!strcmp(argv[i], "--local"))
    {
      useLocal = 1;
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./nbody [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h  --help               Print the message\n");
      printf("      --list               List available devices\n");
      printf("      --device     INDEX   Select device at INDEX\n");
      printf("  -n  --numbodies  N       Run simulation with N bodies\n");
      printf("  -d  --delta      DELTA   Time difference between iterations\n");
      printf("  -s  --softening  SOFT    Force softening factor\n");
      printf("  -i  --iterations ITRS    Run simulation for ITRS iterations\n");
      printf("      --local              Enable use of local memory\n");
      printf("      --wgsize     WGSIZE  Set work-group size to WGSIZE\n");
      printf("      --width      WIDTH   Set window width to WIDTH\n");
      printf("      --height     HEIGHT  Set window height to HEIGHT\n");
      printf("      --2D                 Initialize with 2D points \n");
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

void releaseGraphics()
{
  SDL_GL_DeleteContext(contextGL);
  SDL_DestroyWindow(window);
  SDL_Quit();
}

void runReference(const float *initialPositions,
                  const float *initialVelocities,
                  float *finalPositions)
{
  size_t dataSize     = numBodies*4*sizeof(float);
  float *positionsIn  = malloc(dataSize);
  float *positionsOut = malloc(dataSize);
  float *velocities   = malloc(dataSize);

  memcpy(positionsIn, initialPositions, dataSize);
  memcpy(velocities, initialVelocities, dataSize);

  for (int itr = 0; itr < iterations; itr++)
  {
    for (int i = 0; i < numBodies; i++)
    {
      float ix = positionsIn[i*4 + 0];
      float iy = positionsIn[i*4 + 1];
      float iz = positionsIn[i*4 + 2];
      float iw = positionsIn[i*4 + 3];

      float fx = 0.f;
      float fy = 0.f;
      float fz = 0.f;

      for (int j = 0; j < numBodies; j++)
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
    float *temp  = positionsIn;
    positionsIn  = positionsOut;
    positionsOut = temp;
  }

  memcpy(finalPositions, positionsIn, dataSize);

  free(positionsIn);
  free(positionsOut);
  free(velocities);
}
