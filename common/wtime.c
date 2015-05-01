#if defined(_WIN32) && !defined(__MINGW32__)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h>
#else
#ifdef _OPENMP
#include <omp.h>
#else
#include <sys/time.h>
#endif
#endif

#include <stdlib.h>

double wtime()
{
#if defined(_WIN32) && !defined(__MINGW32__)
  static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

  SYSTEMTIME  system_time;
  FILETIME    file_time;
  uint64_t    time;

  GetSystemTime( &system_time );
  SystemTimeToFileTime( &system_time, &file_time );
  time =  ((uint64_t)file_time.dwLowDateTime )      ;
  time += ((uint64_t)file_time.dwHighDateTime) << 32;

  return ((time - EPOCH) / 10000000.0) + (system_time.wMilliseconds / 1000.0);
#else
#ifdef _OPENMP
   /* Use omp_get_wtime() if we can */
   return omp_get_wtime();
#else
   /* Use a generic timer */
   static int sec = -1;
   struct timeval tv;
   gettimeofday(&tv, NULL);
   if (sec < 0) sec = tv.tv_sec;
   return (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
#endif
#endif
}
