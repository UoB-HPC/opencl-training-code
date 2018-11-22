Development repository for the UoB-HPC 'Advanced HandsOnOpenCL'
material. At some point, this will probably become public, when we
have ironed out the various issues it currently has.


Dependencies
============

OS X
----
- libsdl2

Ubuntu
------
- mesa-common-dev
- libsdl2-dev
- libgl1-mesa-glx

Windows
-------
- Windows 7 or newer
- Visual Studio 2010 or newer
- Any OpenCL driver (e.g. NVIDIA, AMD or Intel)


Compiling
=========

Unix and OS X
-------------

Just type `make`.
There is a Makefile in each project directory, and global ones in the `exercises` and `solutions` directories.
To change the compiler from the default C/C++ compilers set the `CC`/`CXX` variables; for example `make CXX=icpc`.


Notes
=====

NBody solution
--------------

On OS X, when running on the CPU you will need to select `--wgsize 1` at the command line.
We expect 8 incorrect values.

TODO
====
- [ ] Port remaining exercises to C
- [ ] Update Visual Studio solutions to include C exercises (or use CMake?)
- [ ] Add job scripts for Isambard
