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

#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include <device_picker.h>
#include <util.h>

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

int checkOutput(cl_uint *data, cl_uint value)
{
  if (!data)
    return 0;

  int pass = 1;
  for (unsigned i = 0; i < bufferSize/4; i++)
  {
    pass &= (data[i] == value*42+i);
  }
  return pass;
}

void runBenchmark(cl_context context, cl_command_queue queue,
                  cl_kernel fill,
                  cl_mem  d_buffer, // device buffer
                  cl_uint *h_buffer, // host buffer, ignored for zero-copy
                  int zeroCopy)
{
  int pass = 1;
  double transferTime = 0.0;
  double startTime = getCurrentTimeMicroseconds();
  for (cl_uint i = 0; i < iterations; i++)
  {
    // Run fill kernel
    size_t global[] = {bufferSize/4};
    clSetKernelArg(fill, 0, sizeof(cl_mem), &d_buffer);
    clSetKernelArg(fill, 1, sizeof(cl_uint), &i);
    clEnqueueNDRangeKernel(queue, fill, 1, 0, global, NULL, 0, NULL, NULL);
    clFinish(queue);

    double startTransfer = getCurrentTimeMicroseconds();

    if (zeroCopy)
    {
      // Map device buffer to get host pointer
      // **** TODO ****
    }
    else
    {
      // Read data from device buffer to host buffer
      clEnqueueReadBuffer(queue, d_buffer, CL_TRUE, 0, bufferSize, h_buffer, 0, NULL, NULL);
    }
    double endTransfer = getCurrentTimeMicroseconds();

    // Check data
    pass &= checkOutput(h_buffer, i);

    if (zeroCopy)
    {
      // Unmap host pointer
      // **** TODO ****
    }

    transferTime += (endTransfer - startTransfer);
  }
  clFinish(queue);

  // Print stats
  double endTime  = getCurrentTimeMicroseconds();
  double seconds    = (endTime - startTime) * 1e-6;
  double totalBytes = iterations * (double)bufferSize;
  double bandwidth  = (totalBytes / transferTime) * 1e-3;
  if (pass)
  {
    printf("   %6.2lfs", seconds);
    printf("   %7.2lfs", transferTime*1e-6);
    printf("   %8.2lf GB/s", bandwidth);
    printf("\n");
  }
  else
  {
    printf("   -s");
    printf("   -s");
    printf("   - GB/s");
    printf("   FAILED\n");
  }
}

int main(int argc, char *argv[])
{
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
  printf("\nUsing OpenCL device: %s\n", name);
  printf("Buffer size = %u MB\n", bufferSize);
  printf("Iterations = %u\n", iterations);

  cl_bool unifiedMemory;
  clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &unifiedMemory, NULL);
  if (unifiedMemory)
    printf("Device has host-unified memory\n\n");
  else
    printf("Device does not have host-unified memory\n\n");

  // Convert buffer size to bytes
  bufferSize *= 1024*1024;

  cl_int err;
  cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  cl_kernel fill = clCreateKernel(program, "fill", &err);


  printf("Type          Total   Transfer       Bandwidth\n"
         "----------------------------------------------\n");


  // Baseline - using a regular buffer with enqueueReadBuffer
  {
    // Create device buffer
    cl_mem d_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, NULL, NULL);

    // Create host buffer
    cl_uint *h_buffer = malloc(sizeof(cl_uint)*bufferSize/4);

    printf("Baseline ");
    runBenchmark(context, queue, fill, d_buffer, h_buffer, 0);

    free(h_buffer);
  }

  if (unifiedMemory)
  {
    // Create a host-accessible device buffer
    // **** TODO ****

    // No separate host buffer needed

    //printf("Zero-Copy");
    //runBenchmark(context, queue, fill, d_buffer, NULL, 1);
  }
  else
  {
    // Create device buffer
    // **** TODO ****

    // Create a pinned host buffer (using a mapped device buffer)
    // **** TODO ****

    //printf("Pinned   ");
    //runBenchmark(context, queue, fill, d_buffer, h_pinned, 0);

    // Unmap pinned host buffer
    // **** TODO ****
  }

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
        for (unsigned i = 0; i < numDevices; i++)
        {
          char name[MAX_INFO_STRING];
          getDeviceName(devices[i], name);
          printf("%d: %s\n", i, name);
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
    else if (!strcmp(argv[i], "--size") || !strcmp(argv[i], "-s"))
    {
      if (++i >= argc || !parseUInt(argv[i], &bufferSize))
      {
        printf("Invalid buffer size\n");
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
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./transfer [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h  --help               Print the message\n");
      printf("      --list               List available devices\n");
      printf("      --device     INDEX   Select device at INDEX\n");
      printf("  -s  --size       S       Buffer size in MB\n");
      printf("  -i  --iterations ITRS    Number of benchmark iterations\n");
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
