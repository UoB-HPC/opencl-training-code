//
// OpenCL host<->device transfer exercise
//

#include <iomanip>
#include <iostream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <cl.hpp>
#include <device_picker.hpp>
#include <util.hpp>

void parseArguments(int argc, char *argv[]);

// Simulation parameters, with default values.
unsigned deviceIndex   =      0;
unsigned bufferSize    =    256; // Size in MB
unsigned iterations    =     32;

const char *kernel_source =
"kernel void fill(global uint *data, uint value)"
"{"
"  data[get_global_id(0)] = value*42;"
"}";

bool checkOutput(cl_uint *data, cl_uint value)
{
  if (!data)
    return false;

  bool pass = true;
  for (unsigned i = 0; i < bufferSize/4; i++)
  {
    pass &= (data[i] == value*42);
  }
  return pass;
}

void runBenchmark(cl::Context& context, cl::CommandQueue& queue,
                  cl::make_kernel<cl::Buffer, cl_uint> fill,
                  bool useMapped)
{
  // Create buffer
  cl::Buffer d_buffer(context,
                      CL_MEM_READ_WRITE | (useMapped ? CL_MEM_ALLOC_HOST_PTR:0),
                      bufferSize);
  cl_uint *h_buffer = new cl_uint[bufferSize/4];

  // Run benchmark
  bool pass = true;
  util::Timer timer;
  uint64_t readTime = 0;
  uint64_t startTime = timer.getTimeMicroseconds();
  for (cl_uint i = 0; i < iterations; i++)
  {
    // Run fill kernel
    fill(cl::EnqueueArgs(queue, cl::NDRange(bufferSize/4)), d_buffer, i);
    queue.finish();

    // Read and check data
    uint64_t startRead = timer.getTimeMicroseconds();
    if (useMapped)
    {
      void *ptr = NULL;
      ptr = queue.enqueueMapBuffer(d_buffer, CL_TRUE,
                                   CL_MAP_READ, 0, bufferSize);
      pass &= checkOutput((cl_uint*)ptr, i);
      queue.enqueueUnmapMemObject(d_buffer, ptr);
    }
    else
    {
      queue.enqueueReadBuffer(d_buffer, CL_TRUE, 0, bufferSize, h_buffer);
      pass &= checkOutput(h_buffer, i);
    }
    uint64_t endRead = timer.getTimeMicroseconds();
    readTime += (endRead - startRead);
  }
  queue.finish();

  uint64_t endTime  = timer.getTimeMicroseconds();
  double seconds    = (endTime - startTime) * 1e-6;
  double totalBytes = iterations * (double)bufferSize;
  double bandwidth  = (totalBytes / readTime) * 1e-3;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << (useMapped ? "Using mapped" : "Not mapped  ")
            << "  " << std::setw(6) << seconds   << "s"
            << "  " << std::setw(6) << bandwidth << " GB/s"
            << "  " << (pass ? "PASSED" : "FAILED")
            << std::endl;

  if (!useMapped)
    delete[] h_buffer;
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

    std::string name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << std::endl << "Using OpenCL device: " << name << std::endl
              << "Buffer size = " << bufferSize << " MB" << std::endl
              << "Iterations  = " << iterations << std::endl
              << std::endl;

    // Convert buffer size to bytes
    bufferSize *= 1024*1024;

    cl::Context context(device);
    cl::CommandQueue queue(context);
    cl::Program program(context, kernel_source, true);
    cl::make_kernel<cl::Buffer, cl_uint> fill(program, "fill");

    runBenchmark(context, queue, fill, false);
    runBenchmark(context, queue, fill, true);
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
