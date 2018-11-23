//------------------------------------------------------------------------------
//
// Name:       vadd_chain.c
//
// Purpose:    Chain of elementwise addition of three vectors:
//
//                   d = a + b + c
//                   g = d + e + f
//
// HISTORY:    Written by Tim Mattson, December 2009
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated by Tom Deakin, July 2013
//             Updated by Tom Deakin, October 2014
//             Updated by Tom Deakin, November 2018
//
//------------------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

//pick up device type from compiler command line or from
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif


//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Compute the elementwise sum d = a+b+c
//
// input: a, b and c float vectors of length count
//
// output: d float vector of length count holding the sum a + b + c
//

const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   __global float* d,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       d[i] = a[i] + b[i] + c[i];                                      \n" \
"}                                                                      \n" \
"\n";

//------------------------------------------------------------------------------


int main(int argc, char** argv)
{
    int          err;               // error code returned from OpenCL calls

    float*       h_a = (float*) calloc(LENGTH, sizeof(float));       // a vector
    float*       h_b = (float*) calloc(LENGTH, sizeof(float));       // b vector
    float*       h_c = (float*) calloc(LENGTH, sizeof(float));       // c vector
    float*       h_d = (float*) calloc(LENGTH, sizeof(float));       // d vector (result)
    float*       h_e = (float*) calloc(LENGTH, sizeof(float));       // e vector
    float*       h_f = (float*) calloc(LENGTH, sizeof(float));       // f vector
    float*       h_g = (float*) calloc(LENGTH, sizeof(float));       // g vector (result)

	float tmp;
    unsigned int correct;           // number of correct results

    size_t global;                  // global domain size

	char             name[256];
	cl_uint          numPlatforms;
	cl_platform_id  *platforms;
    cl_device_id     device_id;     // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        ko_vadd;       // compute kernel

    cl_mem d_a;                     // device memory used for the input  a vector
    cl_mem d_b;                     // device memory used for the input  b vector
    cl_mem d_c;                     // device memory used for the input  c vector
    cl_mem d_d;                     // device memory used for the input/output d vector
    cl_mem d_e;                     // device memory used for the input  e vector
    cl_mem d_f;                     // device memory used for the input  f vector
    cl_mem d_g;                     // device memory used for the output g vector

    // Fill vectors a and b with random float values
    unsigned i = 0;
    unsigned count = LENGTH;
    for(i = 0; i < count; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
        h_c[i] = rand() / (float)RAND_MAX;
        h_e[i] = rand() / (float)RAND_MAX;
        h_f[i] = rand() / (float)RAND_MAX;
    }

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    // Get all platforms
    platforms = malloc(numPlatforms*sizeof(cl_platform_id));
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    checkError(err, "Getting platforms");

    // Secure a device
    for (i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(platforms[i], DEVICE, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

    if (device_id == NULL)
        checkError(err, "Finding a device");

	clGetDeviceInfo(device_id, CL_DEVICE_NAME, 256, name, NULL);
	printf("Using device: %s\n", name);

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program
    ko_vadd = clCreateKernel(program, "vadd", &err);
    checkError(err, "Creating kernel");

    // Create the arrays in device memory
    d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_a");

    d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_b");

    d_c  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_c");

    d_d  = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_d");

    d_e  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_e");

    d_f  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_f");

    d_g  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_g");

    // Write vectors into compute device memory
    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * count, h_a, 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_a");

    err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * count, h_b, 0, NULL, NULL);
    checkError(err, "Copying h_b to device at d_b");

    err = clEnqueueWriteBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * count, h_c, 0, NULL, NULL);
    checkError(err, "Copying h_c to device at d_c");

    err = clEnqueueWriteBuffer(commands, d_e, CL_TRUE, 0, sizeof(float) * count, h_e, 0, NULL, NULL);
    checkError(err, "Copying h_e to device at d_e");

    err = clEnqueueWriteBuffer(commands, d_f, CL_TRUE, 0, sizeof(float) * count, h_f, 0, NULL, NULL);
    checkError(err, "Copying h_f to device at d_f");

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(ko_vadd, 3, sizeof(cl_mem), &d_d);
    err |= clSetKernelArg(ko_vadd, 4, sizeof(unsigned int), &count);
    checkError(err, "Setting kernel arguments");

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    global = count;
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_d);
    err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_e);
    err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_f);
    err |= clSetKernelArg(ko_vadd, 3, sizeof(cl_mem), &d_g);
    err |= clSetKernelArg(ko_vadd, 4, sizeof(unsigned int), &count);
    checkError(err, "Setting kernel arguments");

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // Wait for the commands to complete before stopping the timer
    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");

    // Read back the results from the compute device
    err = clEnqueueReadBuffer( commands, d_g, CL_TRUE, 0, sizeof(float) * count, h_g, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
    }

    // Test the results
    correct = 0;

    for(i = 0; i < count; i++)
    {
        tmp = h_a[i] + h_b[i] + h_c[i] + h_e[i] + h_f[i]; // assign element i of a+b+c+e+f to tmp
        tmp -= h_g[i];                                    // compute deviation of expected and output result
        if(tmp*tmp < TOL*TOL)        // correct if square deviation is less than tolerance squared
            correct++;
        else {
            printf(" tmp %f h_a %f h_b %f h_c %f h_e %f h_f %f h_g %f\n",tmp, h_a[i], h_b[i], h_c[i], h_e[i], h_f[i], h_g[i]);
        }
    }

    // summarise results
    printf("G = A+B+C+E+F:  %d out of %d results were correct.\n", correct, count);

    // cleanup then shutdown
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseMemObject(d_d);
    clReleaseMemObject(d_e);
    clReleaseMemObject(d_f);
    clReleaseMemObject(d_g);
    clReleaseProgram(program);
    clReleaseKernel(ko_vadd);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    free(h_e);
    free(h_f);
    free(h_g);

#if defined(_WIN32) && !defined(__MINGW32__)
	system("pause");
#endif

    return 0;
}

