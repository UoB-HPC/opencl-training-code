//------------------------------------------------------------------------------
//
// Name:       pi_ocl.cpp
//
// Purpose:    Numeric integration to estimate pi
//
// HISTORY:    Written by Tim Mattson, May 2010
//             Ported to the C++ Wrapper API by Benedict R. Gaster, September 2011
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <device_picker.hpp>
#include <util.hpp>

#define INSTEPS (512*512*512)
#define ITERS (262144)

int main(int argc, char *argv[])
{
    int in_nsteps = INSTEPS;		// default number of steps (updated later to device prefereable)
    int niters = ITERS;				// number of iterations
    int nsteps;
    float step_size;
    ::size_t nwork_groups;
    ::size_t work_group_size = 8;
    float pi_res;

    cl::Buffer d_partial_sums;

    try
    {
        cl_uint deviceIndex = 0;
        parseArguments(argc, argv, &deviceIndex);

        // Get list of devices
        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);

        // Check device index in range
        if (deviceIndex >= numDevices)
        {
          std::cout << "Invalid device index (try '--list')\n";
          return EXIT_FAILURE;
        }

        cl::Device device = devices[deviceIndex];

        std::string name = getDeviceName(device);
        std::cout << "\nUsing OpenCL device: " << name << "\n";

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(device);
        cl::Context context(chosen_device);
        cl::CommandQueue queue(context, device);

        // Create the program object
        cl::Program program(context, util::loadProgram("pi_ocl.cl"), true);

        cl::KernelFunctor<int, float, cl::LocalSpaceArg, cl::Buffer> pi(program, "pi");

        // Get the kernel object for querying information
        cl::Kernel ko_pi = pi.getKernel();

        // Get the work group size
        work_group_size = ko_pi.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        //printf("wgroup_size = %lu\n", work_group_size);

        // Now that we know the size of the work_groups, we can set the number of work
        // groups, the actual number of steps, and the step size
        nwork_groups = in_nsteps/(work_group_size*niters);

        if ( nwork_groups < 1) {
            nwork_groups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            work_group_size=in_nsteps / (nwork_groups*niters);
        }

        nsteps = work_group_size * niters * nwork_groups;
        step_size = 1.0f/static_cast<float>(nsteps);
        std::vector<float> h_psum(nwork_groups);

        printf(
            " %d work groups of size %d.  %d Integration steps\n",
            (int)nwork_groups,
            (int)work_group_size,
            nsteps);

        d_partial_sums = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nwork_groups);

        util::Timer timer;

        // Execute the kernel over the entire range of our 1d input data set
        // using the maximum number of work group items for this device
        pi(
            cl::EnqueueArgs(
                    queue,
                    cl::NDRange(nsteps / niters),
                    cl::NDRange(work_group_size)),
                    niters,
                    step_size,
                    cl::Local(sizeof(float) * work_group_size),
                    d_partial_sums);

        cl::copy(queue, d_partial_sums, h_psum.begin(), h_psum.end());

        // complete the sum and compute final integral value
        pi_res = 0.0f;
        for (unsigned int i = 0; i< nwork_groups; i++) {
                pi_res += h_psum[i];
        }
        pi_res = pi_res * step_size;

        //rtime = wtime() - rtime;
        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.;
        printf("\nThe calculation ran in %lf seconds\n", rtime);
        printf(" pi = %f for %d steps\n", pi_res, nsteps);

    }
    catch (cl::BuildError error)
    {
      std::string log = error.getBuildLog()[0].second;
      std::cerr << std::endl << "Build failed:" << std::endl << log << std::endl;
    }
    catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr
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

}

