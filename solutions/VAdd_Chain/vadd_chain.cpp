//------------------------------------------------------------------------------
//
// Name:       vadd_chain.cpp
//
// Purpose:    Chain of elementwise addition of three vectors:
//
//                   d = a + b + c
//                   g = d + e + f
//                   
//
// HISTORY:    Written by Tim Mattson, June 2011
//             Ported to C++ Wrapper API by Benedict Gaster, September 2011
//             Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//             Updated by Tom Deakin, November 2018
//
//------------------------------------------------------------------------------

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>



#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <util.hpp>


// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#include "err_code.h"

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

int main(void)
{
    std::vector<float> h_a(LENGTH);                     // a vector
    std::vector<float> h_b(LENGTH);                     // b vector
    std::vector<float> h_c(LENGTH);                     // c vector
    std::vector<float> h_d(LENGTH, (float)0xdeadbeef);  // d vector (result)
    std::vector<float> h_e(LENGTH);                     // e vector
    std::vector<float> h_f(LENGTH);                     // f vector
    std::vector<float> h_g(LENGTH, (float)0xdeadbeef);  // g vector (result)

    cl::Buffer d_a;                       // device memory used for the input  a vector
    cl::Buffer d_b;                       // device memory used for the input  b vector
    cl::Buffer d_c;                       // device memory used for the input  c vector
    cl::Buffer d_d;                       // device memory used for the input/output d vector
    cl::Buffer d_e;                       // device memory used for the input  e vector
    cl::Buffer d_f;                       // device memory used for the input  f vector
    cl::Buffer d_g;                       // device memory used for the output g vector

    // Fill input vectors with random float values
    int count = LENGTH;
    for(int i = 0; i < count; i++)
    {
        h_a[i]  = rand() / (float)RAND_MAX;
        h_b[i]  = rand() / (float)RAND_MAX;
        h_c[i]  = rand() / (float)RAND_MAX;
        h_e[i]  = rand() / (float)RAND_MAX;
        h_f[i]  = rand() / (float)RAND_MAX;
    }

    try
    {
    	// Create a context
        cl::Context context(DEVICE);

        cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        std::cout << std::endl << "Using OpenCL device: "
                  << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // Load in kernel source, creating a program object for the context
        cl::Program program(context, util::loadProgram("vadd_chain.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor

        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(program, "vadd");

        d_a   = cl::Buffer(context, h_a.begin(), h_a.end(), true);
        d_b   = cl::Buffer(context, h_b.begin(), h_b.end(), true);
        d_c   = cl::Buffer(context, h_c.begin(), h_c.end(), true);
        d_e   = cl::Buffer(context, h_e.begin(), h_e.end(), true);
        d_f   = cl::Buffer(context, h_f.begin(), h_f.end(), true);

        d_d  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
        d_g  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)),
            d_a,
            d_b,
            d_c,
            d_d,
            count);

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)),
            d_d,
            d_e,
            d_f,
            d_g,
            count);

        cl::copy(queue, d_g, h_g.begin(), h_g.end());

        // Test the results
        int correct = 0;
        float tmp;
        for(int i = 0; i < count; i++)
        {
            tmp = h_a[i] + h_b[i] + h_c[i] + h_e[i] + h_f[i]; // assign element i of a+b+c+e+f to tmp
            tmp -= h_g[i];                                    // compute deviation of expected and output result
            if(tmp*tmp < TOL*TOL)                             // correct if square deviation is less than tolerance squared
                correct++;
            else {
                printf(" tmp %f h_a %f h_b %f h_c %f h_e %f h_f %f h_g %f\n",tmp, h_a[i], h_b[i], h_c[i], h_e[i], h_f[i], h_g[i]);
            }
        }

        // summarize results
        printf("G = A+B+C+E+F:  %d out of %d results were correct.\n", correct, count);

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
