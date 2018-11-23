/*
 *
 * This code is released under the "attribution CC BY" creative commons license.
 * In other words, you can use it in any way you see fit, including commercially,
 * but please retain an attribution for the original authors:
 * the High Performance Computing Group at the University of Bristol.
 * Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
 *
 */

#ifndef __UTIL_HDR
#define __UTIL_HDR

#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32) && !defined(__MINGW32__)
#include <time.h>
#else
#include <sys/time.h>
#endif

// Utility to load an OpenCL kernel source file
char *loadProgram(const char *filename)
{
  FILE *file = fopen(filename, "rb");
  if (!file)
  {
    fprintf(stderr, "Error: Could not open kernel source file '%s'\n", filename);
    exit(EXIT_FAILURE);
  }

  fseek(file, 0, SEEK_END);
  int len = ftell(file) + 1;
  rewind(file);

  char *source = (char *)calloc(sizeof(char), len);
  if (!source)
  {
     fprintf(stderr, "Error: Could not allocate memory for source string\n");
     exit(EXIT_FAILURE);
  }
  fread(source, sizeof(char), len, file);
  fclose(file);
  return source;
}

// Utility to return the current time in nanoseconds since the epoch
double getCurrentTimeNanoseconds()
{
#if defined(_WIN32) && !defined(__MINGW32__)
  return time(NULL) * 1e9;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec * 1e3 + tv.tv_sec * 1e9;
#endif
}

// Utility to return the current time in microseconds since the epoch
double getCurrentTimeMicroseconds()
{
  return getCurrentTimeNanoseconds() / 1e3;
}

// Utility to return the current time in seconds since the epoch
double wtime()
{
  return getCurrentTimeNanoseconds() / 1e9;
}

#endif
