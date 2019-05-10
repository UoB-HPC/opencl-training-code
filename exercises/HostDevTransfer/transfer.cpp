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

#include <iomanip>
#include <iostream>
#include <vector>

#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#endif
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <device_picker.hpp>
#include <util.hpp>

void parseArguments(int argc, char *argv[]);

// Benchmark parameters, with default values.
unsigned deviceIndex   =      0;
unsigned bufferSize    =      2; // Size in MB
unsigned iterations    =     32;

const char *kernel_source =
"kernel void fill(global uint *data, uint value)"
"{"
"  uint i = get_global_id(0);"
"  data[i] = value*42 + i;"
"}";

bool checkOutput(cl_uint *data, cl_uint value)
{
  if (!data)
    return false;

  bool pass = true;
  for (unsigned i = 0; i < bufferSize/4; i++)
  {
    pass &= (data[i] == value*42+i);
  }
  return pass;
}

void runBenchmark(cl::Context& context, cl::CommandQueue& queue,
                  cl::KernelFunctor<cl::Buffer, cl_uint> fill,
                  cl::Buffer& d_buffer, // device buffer
                  cl_uint    *h_buffer, // host buffer, ignored for zero-copy
                  bool zeroCopy)
{
  bool pass = true;
  util::Timer timer;
  uint64_t transferTime = 0;
  uint64_t startTime = timer.getTimeMicroseconds();
  for (cl_uint i = 0; i < iterations; i++)
  {
    // Run fill kernel
    fill(cl::EnqueueArgs(queue, cl::NDRange(bufferSize/4)), d_buffer, i);
    queue.finish();

    uint64_t startTransfer = timer.getTimeMicroseconds();

    if (zeroCopy)
    {
      // Map device buffer to get host pointer
      // **** TODO ****
    }
    else
    {
      // Read data from device buffer to host buffer
      queue.enqueueReadBuffer(d_buffer, CL_TRUE, 0, bufferSize, h_buffer);
    }
    uint64_t endTransfer = timer.getTimeMicroseconds();

    // Check data
    pass &= checkOutput(h_buffer, i);

    if (zeroCopy)
    {
      // Unmap host pointer
      // **** TODO ****
    }

    transferTime += (endTransfer - startTransfer);
  }
  queue.finish();

  // Print stats
  uint64_t endTime  = timer.getTimeMicroseconds();
  double seconds    = (endTime - startTime) * 1e-6;
  double totalBytes = iterations * (double)bufferSize;
  double bandwidth  = (totalBytes / transferTime) * 1e-3;
  std::cout << std::fixed << std::setprecision(2);
  if (pass)
  {
    std::cout << "   " << std::setw(6) << seconds   << "s"
              << "   " << std::setw(7) << transferTime*1e-6 << "s"
              << "   " << std::setw(8) << bandwidth << " GB/s"
              << std::endl;
  }
  else
  {
    std::cout << "   " << std::setw(6) << "-" << "s"
              << "   " << std::setw(7) << "-" << "s"
              << "   " << std::setw(8) << "-" << " GB/s"
              << "   FAILED"
              << std::endl;
  }
}

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
              << "Buffer size = " << bufferSize << " MB" << std::endl
              << "Iterations  = " << iterations << std::endl;

    bool unifiedMemory = device.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>();
    std::cout << (unifiedMemory ?
      "Device has host-unified memory" :
      "Device does not have host-unified memory")
              << std::endl
              << std::endl;

    // Convert buffer size to bytes
    bufferSize *= 1024*1024;

    cl::Context context(device);
    cl::CommandQueue queue(context);
    cl::Program program(context, kernel_source, true);
    cl::KernelFunctor<cl::Buffer, cl_uint> fill(program, "fill");

    std::cout << "Type          Total   Transfer       Bandwidth" << std::endl
              << "----------------------------------------------" << std::endl;


    // Baseline - using a regular buffer with enqueueReadBuffer
    {
      // Create device buffer
      cl::Buffer d_buffer(context, CL_MEM_READ_WRITE, bufferSize);

      // Create host buffer
      cl_uint *h_buffer = new cl_uint[bufferSize/4];

      std::cout << "Baseline ";
      runBenchmark(context, queue, fill, d_buffer, h_buffer, false);

      delete[] h_buffer;
    }

    if (unifiedMemory)
    {
      // Create a host-accessible device buffer
      // **** TODO ****

      // No separate host buffer needed

      //std::cout << "Zero-Copy";
      //runBenchmark(context, queue, fill, d_buffer, NULL, true);
    }
    else
    {
      // Create device buffer
      // **** TODO ****

      // Create a pinned host buffer (using a mapped device buffer)
      // **** TODO ****

      //std::cout << "Pinned   ";
      //runBenchmark(context, queue, fill, d_buffer, h_pinned, false);

      // Unmap pinned host buffer
      // **** TODO ****
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
    else if (!strcmp(argv[i], "--size") || !strcmp(argv[i], "-s"))
    {
      if (++i >= argc || !parseUInt(argv[i], &bufferSize))
      {
        std::cout << "Invalid buffer size" << std::endl;
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
      std::cout << "Usage: ./transfer [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help               Print the message" << std::endl;
      std::cout << "      --list               List available devices" << std::endl;
      std::cout << "      --device     INDEX   Select device at INDEX" << std::endl;
      std::cout << "  -s  --size       S       Buffer size in MB" << std::endl;
      std::cout << "  -i  --iterations ITRS    Number of benchmark iterations" << std::endl;
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
