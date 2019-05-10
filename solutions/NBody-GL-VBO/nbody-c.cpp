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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#define GLEW_STATIC
#include <GL/glew.h>
#endif

#if !defined(_WIN32) && !defined(__APPLE__)
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glext.h>
#endif

#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#endif

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#undef main

#if defined(__APPLE__)
  #include <OpenCL/OpenCL.h>
#else
  #define CL_TARGET_OPENCL_VERSION 120
  #include <CL/cl.h>
  #include <CL/cl_ext.h>
  #include <CL/cl_gl.h>
#endif

#include <device_picker.h>
#include <util.h>
#include <util.hpp>

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

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
cl_float delta         =      0.0001f;
cl_float softening     =      0.05f;
cl_uint  iterations    =     16;
float    sphereRadius  =      0.8f;
float    tolerance     =      0.01f;
unsigned wgsize        =     16;
int      useLocal      =      0;
unsigned init2D        =      0;
cl_uint  windowWidth   =    640;
cl_uint  windowHeight  =    480;

// SDL/GL objects
SDL_Window    *window;
SDL_GLContext  contextGL;
static struct
{
  GLuint program;
  GLuint positions[2];
} gl;

int main(int argc, char *argv[])
{
  cl_int           err;
  cl_device_id     device;
  cl_context       context;
  cl_command_queue queue;
  cl_program       program;
  cl_kernel        nbodyKernel;
  double           start, end;

  parseArguments(argc, argv);

  initGraphics();

  // Initialize host data
  size_t dataSize            = numBodies*sizeof(cl_float4);
  float *h_initialPositions  = (float*)malloc(dataSize);
  float *h_initialVelocities = (float*)malloc(dataSize);
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

  char extensions[1024];
  err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1024, extensions, NULL);
  checkError(err, "getting device extensions");

#if defined(__APPLE__)
  bool useGLInterop = strstr(extensions, "cl_APPLE_gl_sharing") != NULL;
#else
  bool useGLInterop = strstr(extensions, "cl_khr_gl_sharing") != NULL;
#endif

  char name[MAX_INFO_STRING];
  getDeviceName(device, name);
  printf("\nUsing OpenCL device: %s\n", name);
  if (!useGLInterop)
    printf("WARNING: CL/GL not supported\n");

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

  context = clCreateContext(
    useGLInterop ? properties : NULL, 1, &device, NULL, NULL, &err);
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
    char *buildLog = (char*)malloc(++sz);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sz, buildLog, NULL);
    fprintf(stderr, "%s\n", buildLog);
    free(buildLog);
  }
  checkError(err, "building program");

  nbodyKernel = clCreateKernel(program, "nbody", &err);
  checkError(err, "creating nbody kernel");

  // Initialize device buffers
  cl_mem d_positions[2], d_velocities;

  if (useGLInterop)
  {
    d_positions[0] = clCreateFromGLBuffer(
      context, CL_MEM_READ_WRITE, gl.positions[0], &err);
    checkError(err, "creating d_positions[0] buffer");
    d_positions[1] = clCreateFromGLBuffer(
      context, CL_MEM_READ_WRITE, gl.positions[1], &err);
    checkError(err, "creating d_positions[1] buffer");
  }
  else
  {
    d_positions[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  dataSize, NULL, &err);
    checkError(err, "creating d_positions[0] buffer");

    d_positions[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  dataSize, NULL, &err);
    checkError(err, "creating d_positions[1] buffer");
  }

  d_velocities = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                dataSize, NULL, &err);
  checkError(err, "creating d_velocities buffer");

  if (useGLInterop)
  {
    // We have to acquire the objects from GL before we can copy
    glFinish();
    err = clEnqueueAcquireGLObjects(queue, 2, d_positions, 0, NULL, NULL);
    checkError(err, "acquire GL objects");
    err = clEnqueueWriteBuffer(queue, d_positions[0], CL_TRUE,
                               0, dataSize, h_initialPositions, 0, NULL, NULL);
    checkError(err, "writing d_positions data");
    err = clEnqueueReleaseGLObjects(queue, 2, d_positions, 0, NULL, NULL);
    checkError(err, "release GL objects");
    err = clFinish(queue);
    checkError(err, "finishing queue");
  }
  else
  {
    err = clEnqueueWriteBuffer(queue, d_positions[0], CL_TRUE,
                               0, dataSize, h_initialPositions, 0, NULL, NULL);
    checkError(err, "writing d_positions data");
  }

  err = clEnqueueWriteBuffer(queue, d_velocities, CL_TRUE,
                             0, dataSize, h_initialVelocities, 0, NULL, NULL);
  checkError(err, "writing d_velocities data");

  err  = clSetKernelArg(nbodyKernel, 2, sizeof(cl_mem), &d_velocities);
  err |= clSetKernelArg(nbodyKernel, 3, sizeof(cl_uint), &numBodies);
  checkError(err, "setting nbody kernel args");

  printf("OpenCL initialization complete.\n\n");


  // Run simulation
  unsigned INDEX_IN  = 0;
  unsigned INDEX_OUT = 1;
  printf("Running simulation...\n");
  start = getCurrentTimeNanoseconds();
  size_t global[1]      = {numBodies};
  size_t local[1]       = {wgsize};
  size_t i;
  for (i = 0; ; i++)
  {
    // Flush GL queue and acquire buffers
    glFlush();
    if (useGLInterop)
    {
      err = clEnqueueAcquireGLObjects(queue, 2, d_positions, 0, NULL, NULL);
      checkError(err, "acquiring GL objects");
    }

    // Enqueue nbody kernel
    err  = clSetKernelArg(nbodyKernel, 0, sizeof(cl_mem),
                          &d_positions[INDEX_IN]);
    err |= clSetKernelArg(nbodyKernel, 1, sizeof(cl_mem),
                          &d_positions[INDEX_OUT]);
    checkError(err, "setting kernel arguments");
    err = clEnqueueNDRangeKernel(queue, nbodyKernel,
                                  1, NULL, global, local, 0, NULL, NULL);
    checkError(err, "enqueuing nbody kernel");

    // Release buffers
    if (useGLInterop)
    {
      err = clEnqueueReleaseGLObjects(queue, 2, d_positions, 0, NULL, NULL);
      checkError(err, "releasing GL objects");
    }

    // Finish CL queue
    err = clFinish(queue);
    checkError(err, "finishing CL queue");

    // Manually copy data into GL vertex buffer if we don't have GL interop
    if (!useGLInterop)
    {
      void *data = clEnqueueMapBuffer(queue, d_positions[INDEX_OUT],
                                          CL_TRUE, CL_MAP_READ,
                                          0, numBodies*sizeof(cl_float4),
                                          0, NULL, NULL, &err);
      checkError(err, "mapping buffer");
      glBufferSubData(GL_ARRAY_BUFFER, 0, numBodies*sizeof(cl_float4), data);

      err = clEnqueueUnmapMemObject(queue, d_positions[INDEX_OUT], data,
                                    0, NULL, NULL);
      checkError(err, "unmapping buffer");
    }

    // Render body positions
    glUseProgram(gl.program);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    GLint loc = glGetAttribLocation(gl.program, "positions");
    glBindBuffer(GL_ARRAY_BUFFER, gl.positions[INDEX_OUT]);
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_POINTS, 0, numBodies);

    // Update window
    SDL_GL_SwapWindow(window);

    // Check for user input
    if (handleSDLEvents())
    {
      break;
    }

    // Swap buffers
    int temp  = INDEX_IN;
    INDEX_IN  = INDEX_OUT;
    INDEX_OUT = temp;
  }

  end = getCurrentTimeNanoseconds();
  printf("OpenCL took %.2lfms\n\n", (end-start)*1e-6);

  printf("Average FPS was %.1f\n\n", i / ((end-start)*1e-9));

  free(h_initialPositions);
  free(h_initialVelocities);
  clReleaseMemObject(d_positions[0]);
  clReleaseMemObject(d_positions[1]);
  clReleaseMemObject(d_velocities);
  clReleaseKernel(nbodyKernel);
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

  #if defined(_WIN32)
    glewInit();
  #endif

  // Build vertex shader
  std::string vert_shader = util::loadProgram("vert_shader.glsl");
  const char *vert_shader_glsl = vert_shader.c_str();
  GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertShader, 1, &vert_shader_glsl, NULL);
  glCompileShader(vertShader);
  {
    GLint buildResult = GL_FALSE;
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &buildResult);
    int   buildLogLen = 0;
    glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &buildLogLen);
    char  *buildLog = new char[buildLogLen];
	memset(buildLog, 0, buildLogLen);
    glGetShaderInfoLog(vertShader, buildLogLen, NULL, buildLog);
    if (GL_TRUE != buildResult)
    {
      fprintf(stderr, "Error whilst building vertex shader: \n%s\n", buildLog);
    }
    delete[] buildLog;
  }

  // Build fragment shader
  std::string frag_shader = util::loadProgram("frag_shader.glsl");
  const char *frag_shader_glsl = frag_shader.c_str();
  GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragShader, 1, &frag_shader_glsl, NULL);
  glCompileShader(fragShader);
  {
    GLint buildResult = GL_FALSE;
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &buildResult);
    int   buildLogLen = 0;
    glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &buildLogLen);
    char *buildLog = new char[buildLogLen];
    memset(buildLog, 0, buildLogLen);
    glGetShaderInfoLog(fragShader, buildLogLen, NULL, buildLog);
    if (GL_TRUE != buildResult)
    {
      fprintf(stderr, "Error whilst building fragment shader: \n%s\n", buildLog);
    }
    delete[] buildLog;
  }

  // Create progam
  gl.program = glCreateProgram();
  glAttachShader(gl.program, vertShader);
  glAttachShader(gl.program, fragShader);
  glLinkProgram(gl.program);
  {
    GLint linkResult = GL_FALSE;
    glGetProgramiv(gl.program, GL_LINK_STATUS, &linkResult);
    int   linkLogLen = 0;
    glGetProgramiv(gl.program, GL_INFO_LOG_LENGTH, &linkLogLen);
    char *linkLog = new char[linkLogLen];
    memset(linkLog, 0, linkLogLen);
    glGetProgramInfoLog(gl.program, linkLogLen, NULL, linkLog);
    if (GL_TRUE != linkResult)
    {
      fprintf(stderr, "Unable to link shaders:\n%s\n", linkLog);
      exit(1);
    }
    delete[] linkLog;
  }

  glDeleteShader(vertShader);
  glDeleteShader(fragShader);


  glEnable(GL_POINT_SPRITE);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);


  // Generate view matrix
  glm::vec3 eye = glm::vec3(0, 0, 2);
  glm::vec3 target = glm::vec3(0, 0, -1000);
  glm::vec3 up = glm::vec3(0, 1, 0);
  glm::mat4 viewMatrix = glm::lookAt(eye, target, up);

  // Generate projection matrix
  float fov          = 2.0f * atan(1.0f / eye.z);
  float aspectRatio  = windowWidth/(float)windowHeight;
  float nearPlane    = 0.1f;
  float farPlane     = 50.f;
  glm::mat4 projMatrix = glm::perspective(fov, aspectRatio, nearPlane, farPlane);

  glm::mat4 vpMatrix = projMatrix * viewMatrix;

  glUseProgram(gl.program);
  glUniformMatrix4fv(glGetUniformLocation(gl.program, "vpMatrix"),
                     1, GL_FALSE, &vpMatrix[0][0]);
  glUniform3fv(glGetUniformLocation(gl.program, "eyePosition"), 1, &eye[0]);
  glUniform1f(glGetUniformLocation(gl.program, "pointScale"), 20.f);
  glUniform1f(glGetUniformLocation(gl.program, "sightRange"), 3.f);


  // Create buffers
  glGenBuffers(2, gl.positions);

  glBindBuffer(GL_ARRAY_BUFFER, gl.positions[0]);
  glBufferData(GL_ARRAY_BUFFER, numBodies*sizeof(cl_float4),
               NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, gl.positions[1]);
  glBufferData(GL_ARRAY_BUFFER, numBodies*sizeof(cl_float4),
               NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);


  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
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
  float *positionsIn  = (float*)malloc(dataSize);
  float *positionsOut = (float*)malloc(dataSize);
  float *velocities   = (float*)malloc(dataSize);

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
