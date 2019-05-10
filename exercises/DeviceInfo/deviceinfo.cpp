/*
 * Display Device Information
 *
 * Script to print out some information about the OpenCL devices
 * and platforms available on your system
 *
 * History: C++ version written by Tom Deakin, 2012
 *          Updated by Tom Deakin, August 2013
*/

/*
 *
 * This code is released under the "attribution CC BY" creative commons license.
 * In other words, you can use it in any way you see fit, including commercially,
 * but please retain an attribution for the original authors:
 * the High Performance Computing Group at the University of Bristol.
 * Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
 *
 */

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

int main(void)
{
  try
  {
    // Discover number of platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::cout << "\nNumber of OpenCL plaforms: " << platforms.size() << std::endl;

    // Investigate each platform
    std::cout << "\n-------------------------" << std::endl;
    for (std::vector<cl::Platform>::iterator plat = platforms.begin(); plat != platforms.end(); plat++)
    {
      std::string s;
      plat->getInfo(CL_PLATFORM_NAME, &s);
      std::cout << "Platform: " << s << std::endl;

      plat->getInfo(CL_PLATFORM_VENDOR, &s);
      std::cout << "\tVendor:  " << s << std::endl;

      plat->getInfo(CL_PLATFORM_VERSION, &s);
      std::cout << "\tVersion: " << s << std::endl;

      // Discover number of devices
      std::vector<cl::Device> devices;
      plat->getDevices(CL_DEVICE_TYPE_ALL, &devices);
      std::cout << "\n\tNumber of devices: " << devices.size() << std::endl;

      // Investigate each device
      for (std::vector<cl::Device>::iterator dev = devices.begin(); dev != devices.end(); dev++ )
      {
        std::cout << "\t-------------------------" << std::endl;

        std::cout << "\t\tName: " << getDeviceName(*dev) << std::endl;

        dev->getInfo(CL_DEVICE_OPENCL_C_VERSION, &s);
        std::cout << "\t\tVersion: " << s << std::endl;

        cl_uint i;
        dev->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &i);
        std::cout << "\t\tMax. Compute Units: " << i << std::endl;

        cl_ulong size;
        dev->getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
        std::cout << "\t\tLocal Memory Size: " << size/1024 << " KB" << std::endl;

        dev->getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &size);
        std::cout << "\t\tGlobal Memory Size: " << size/(1024*1024) << " MB" << std::endl;

        dev->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);
        std::cout << "\t\tMax Alloc Size: " << size/(1024*1024) << " MB" << std::endl;

        size_t maxWGSize;
        dev->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWGSize);
        std::cout << "\t\tMax Work-group Total Size: " << maxWGSize << std::endl;

        std::vector<size_t> d;
        dev->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &d);
        std::cout << "\t\tMax Work-group Dims: (";
        for (std::vector<size_t>::iterator st = d.begin(); st != d.end(); st++)
          std::cout << *st << " ";
        std::cout << "\x08)" << std::endl;

        std::cout << "\t-------------------------" << std::endl;

      }

      std::cout << "\n-------------------------\n";
    }

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

#if defined(_WIN32) && !defined(__MINGW32__)
  system("pause");
#endif

  return 0;
}
