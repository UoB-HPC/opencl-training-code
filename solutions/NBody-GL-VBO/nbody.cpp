//
// OpenCL NBody example
//

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#if defined(_WIN32)
  #define GLEW_STATIC
  #include <GL/glew.h>
#endif

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#undef main

#define __CL_ENABLE_EXCEPTIONS
#include <cl.hpp>

#include "util.hpp"
#include "err_code.h"
#include "device_picker.hpp"

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
void     runReference(const std::vector<float>& initialPositions,
                      const std::vector<float>& initialVelocities,
                            std::vector<float>& finalPositions);

// Simulation parameters, with default values.
cl_uint  deviceIndex   =      0;
cl_uint  numBodies     =   4096;
cl_float delta         =      0.0001f;
cl_float softening     =      0.05f;
cl_uint  iterations    =     16;
float    sphereRadius  =    0.8f;
float    tolerance     =      0.01f;
unsigned unrollFactor  =      1;
unsigned wgsize        =     16;
unsigned init2D        =      0;
cl_uint  windowWidth   =    640;
cl_uint  windowHeight  =    480;

// SDL/GL objects
SDL_Window    *window;
SDL_GLContext  contextGL;
struct
{
  GLuint program;
  GLuint positions[2];
} gl;

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
    std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
    bool useGLInterop = extensions.find("cl_khr_gl_sharing") != std::string::npos;

    std::string name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << std::endl << "Using OpenCL device: " << name << std::endl;
    if (!useGLInterop)
      std::cout << "WARNING: CL/GL not supported" << std::endl;

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


    cl::Context context(device, useGLInterop ? properties : NULL);
    cl::CommandQueue queue(context);

    cl::Program program(context, util::loadProgram("kernel.cl"));
    try
    {
      std::stringstream options;
      options.setf(std::ios::fixed, std::ios::floatfield);
      options << " -cl-fast-relaxed-math";
      options << " -cl-single-precision-constant";
      options << " -Dsoftening=" << softening << "f";
      options << " -Ddelta=" << delta << "f";
      options << " -DUNROLL_FACTOR=" << unrollFactor;
      options << " -DWGSIZE=" << wgsize;
      program.build(options.str().c_str());
    }
    catch (cl::Error error)
    {
      if (error.err() == CL_BUILD_PROGRAM_FAILURE)
      {
        std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << log << std::endl;
      }
      throw(error);
    }

    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl_uint>
      nbodyKernel(program, "nbody");

    // Initialize device buffers
    cl::Buffer d_positions[2], d_velocities;

    if (useGLInterop)
    {
      d_positions[0] = cl::BufferGL(context, CL_MEM_READ_WRITE, gl.positions[0]);
      d_positions[1] = cl::BufferGL(context, CL_MEM_READ_WRITE, gl.positions[1]);
    }
    else
    {
      d_positions[0] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 4 * numBodies*sizeof(cl_float));
      d_positions[1] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 4 * numBodies*sizeof(cl_float));
    }
    d_velocities = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              4*numBodies*sizeof(float));

    cl::copy(queue, begin(h_initialPositions), end(h_initialPositions),
             d_positions[0]);
    cl::copy(queue, begin(h_initialVelocities), end(h_initialVelocities),
             d_velocities);

    std::vector<cl::Memory> clglObjects;
    clglObjects.push_back(d_positions[0]);
    clglObjects.push_back(d_positions[1]);

    std::cout << "OpenCL initialization complete." << std::endl << std::endl;


    // Run simulation
    unsigned INDEX_IN  = 0;
    unsigned INDEX_OUT = 1;
    std::cout << "Running simulation..." << std::endl;
    startTime = timer.getTimeMicroseconds();
    cl::NDRange global(numBodies);
    cl::NDRange local(wgsize);
    cl::NDRange textureSize(windowWidth, windowHeight);
    size_t i;
    for (i = 0; ; i++)
    {
      // ***********************
      // Acquire buffers from GL
      // ***********************
      glFlush();
      if (useGLInterop)
        queue.enqueueAcquireGLObjects(&clglObjects);

      nbodyKernel(cl::EnqueueArgs(queue, global, local),
                  d_positions[INDEX_IN], d_positions[INDEX_OUT], d_velocities,
                  numBodies);

      // **************************
      // Release buffers back to GL
      // **************************
      if (useGLInterop)
        queue.enqueueReleaseGLObjects(&clglObjects);
      queue.flush();

      // Manually copy data into GL vertex buffer if we don't have GL interop
      if (!useGLInterop)
      {
        void *data = queue.enqueueMapBuffer(d_positions[INDEX_OUT],
                                            CL_TRUE, CL_MAP_READ,
                                            0, numBodies*sizeof(cl_float4));
        glBufferSubData(GL_ARRAY_BUFFER, 0, numBodies*sizeof(cl_float4), data);
        queue.enqueueUnmapMemObject(d_positions[INDEX_OUT], data);
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

    endTime = timer.getTimeMicroseconds();
    std::cout << "OpenCL took " << ((endTime-startTime)*1e-3) << "ms"
              << std::endl << std::endl;

    std::cout << "Average FPS was " << (i / ((endTime-startTime)*1e-6))
              << std::endl << std::endl;

    releaseGraphics();
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
          std::string name = devices[i].getInfo<CL_DEVICE_NAME>();
          std::cout << i << ": " << name << std::endl;
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
    else if (!strcmp(argv[i], "--unroll") || !strcmp(argv[i], "-u"))
    {
      if (++i >= argc || !parseUInt(argv[i], &unrollFactor))
      {
        std::cout << "Invalid unroll factor" << std::endl;
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
      std::cout << "  -u  --unroll     UNROLL  Unroll factor" << std::endl;
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
