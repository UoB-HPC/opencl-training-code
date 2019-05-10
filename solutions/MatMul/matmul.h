//------------------------------------------------------------------------------
//
//  Include fle for the Matrix Multiply test harness
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Ported to C by Tom Deakin, July 2013
//
//------------------------------------------------------------------------------

#ifndef __MULT_HDR
#define __MULT_HDR

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

#include "matrix_lib.h"

//------------------------------------------------------------------------------
//  functions from ../Common
//------------------------------------------------------------------------------
extern int    output_device_info(cl_device_id );
extern double wtime();   // returns time since some fixed past point (wtime.c)

//------------------------------------------------------------------------------
//  Constants
//------------------------------------------------------------------------------
#define ORDER    1024    // Order of the square matrices A, B, and C
#define AVAL     3.0     // A elements are constant and equal to AVAL
#define BVAL     5.0     // B elements are constant and equal to BVAL
#define TOL      (0.001) // tolerance used in floating point comparisons
#define DIM      2       // Max dim for NDRange
#define COUNT    1       // number of times to do each multiplication
#define SUCCESS  1
#define FAILURE  0

// Block size for blocked matrix multiply
// Default value can be override at compile-time with -DBLOCKSIZE=X
#ifndef BLOCKSIZE
#define BLOCKSIZE 8
#endif

#endif
