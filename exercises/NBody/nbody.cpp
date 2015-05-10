//
// OpenCL NBody example
//

#include <cmath>
#include <iostream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <cl.hpp>

#include "util.hpp"
#include "err_code.h"
#include "device_picker.hpp"

#ifndef M_PI
  #define M_PI 3.14159265358979323846f
#endif

void     parseArguments(int argc, char *argv[]);
void     runReference(const std::vector<float>& initialPositions,
                      const std::vector<float>& initialVelocities,
                            std::vector<float>& finalPositions);

// Simulation parameters, with default values.
cl_uint  deviceIndex   =      0;
cl_uint  numBodies     =   4096;
cl_float delta         =      0.0002f;
cl_float softening     =      0.05f;
cl_uint  iterations    =     32;
float    sphereRadius  =    0.8f;
float    tolerance     =      0.01f;

int main(int argc, char *argv[])
{
  try
  {
    uint64_t startTime, endTime;
    util::Timer timer;

    parseArguments(argc, argv);

    // Initialize host data
    std::vector<float> h_initialPositions(4*numBodies);
    std::vector<float> h_initialVelocities(4*numBodies, 0);
    std::vector<float> h_positions(4*numBodies);
    for (unsigned i = 0; i < numBodies; i++)
    {
      // Generate a random point on the surface of a sphere
      float longitude             = 2.f * M_PI * (rand() / (float)RAND_MAX);
      float latitude              = acos((2.f * (rand() / (float)RAND_MAX)) - 1);
      h_initialPositions[i*4 + 0] = sphereRadius * sin(latitude) * cos(longitude);
      h_initialPositions[i*4 + 1] = sphereRadius * sin(latitude) * sin(longitude);
      h_initialPositions[i*4 + 2] = sphereRadius * cos(latitude);
      h_initialPositions[i*4 + 3] = 1;
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

    cl::Context context(device);
    cl::CommandQueue queue(context);

    cl::Program program(context, util::loadProgram("kernel.cl"));
    try
    {
      program.build();
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

    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl_uint, cl_float, cl_float>
      nbodyKernel(program, "nbody");

    // Initialize device buffers
    cl::Buffer d_positions0, d_positions1, d_velocities;

    d_positions0 = cl::Buffer(context, CL_MEM_READ_WRITE,
                              4*numBodies*sizeof(float));

    d_positions1 = cl::Buffer(context, CL_MEM_READ_WRITE,
                              4*numBodies*sizeof(float));

    d_velocities = cl::Buffer(context, CL_MEM_READ_WRITE,
                              4*numBodies*sizeof(float));

    cl::copy(queue, h_initialPositions.begin(), h_initialPositions.end(),
             d_positions0);
    cl::copy(queue, h_initialVelocities.begin(), h_initialVelocities.end(),
             d_velocities);

    cl::Buffer d_positionsIn  = d_positions0;
    cl::Buffer d_positionsOut = d_positions1;

    std::cout << "OpenCL initialization complete." << std::endl << std::endl;


    // Run simulation
    std::cout << "Running simulation..." << std::endl;
    startTime = timer.getTimeMicroseconds();
    cl::NDRange global(numBodies);
    for (unsigned i = 0; i < iterations; i++)
    {
      nbodyKernel(cl::EnqueueArgs(queue, global),
                  d_positionsIn, d_positionsOut, d_velocities,
                  numBodies, delta, softening);

      // Swap position buffers
      cl::Buffer temp = d_positionsIn;
      d_positionsIn   = d_positionsOut;
      d_positionsOut  = temp;
    }

    // Read final positions
    cl::copy(queue, d_positionsIn, h_positions.begin(), h_positions.end());

    endTime = timer.getTimeMicroseconds();
    std::cout << "OpenCL took " << ((endTime-startTime)*1e-3) << "ms"
              << std::endl << std::endl;


    // Run reference code
    std::cout << "Running reference..." << std::endl;
    startTime = timer.getTimeMicroseconds();
    std::vector<float> h_reference(4*numBodies);
    runReference(h_initialPositions, h_initialVelocities, h_reference);
    endTime = timer.getTimeMicroseconds();
    std::cout << "Reference took " << ((endTime-startTime)*1e-3) << "ms"
              << std::endl << std::endl;


    // Verify final positions
    unsigned errors = 0;
    for (unsigned i = 0; i < numBodies; i++)
    {
      float ix = h_positions[i*4 + 0];
      float iy = h_positions[i*4 + 1];
      float iz = h_positions[i*4 + 2];

      float rx = h_reference[i*4 + 0];
      float ry = h_reference[i*4 + 1];
      float rz = h_reference[i*4 + 2];

      float dx    = (rx-ix);
      float dy    = (ry-iy);
      float dz    = (rz-iz);
      float dist  = sqrt(dx*dx + dy*dy + dz*dz);

      if (dist > tolerance || (dist!=dist))
      {
        if (!errors)
        {
          std::cout << "Verification failed:" << std::endl;
        }

        // Only show the first 8 errors
        if (errors++ < 8)
        {
          std::cout << "-> Position error at " << i << ": " << dist << std::endl;
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
    std::cout << std::endl;
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
        float invdist = 1.f / dist;
        float coeff = jw * (invdist*invdist*invdist);
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
