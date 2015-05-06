Development repository for the UoB-HPC 'Advanced HandsOnOpenCL'
material. At some point, this will probably become public, when we
have ironed out the various issues it currently has.


Managing Slides
===============
Although PowerPoint doesn't play well with Git, it's still useful to
track versions and have a history of who changed what and
when. PowerPoint for Mac has a reasonable 'compare and merge' facility
(under the 'Review' tab), which can be used if you encounter a merge
conflict. Email-based mutexes could also be used to reduce the risk of
conflicts.


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
To change the compiler from the default C++ compiler set the `CXX` variable; for example `make CXX=icpc`.


TODO
====
- [x] Merge pre-existing conflict for 'part2' slides (there's an 'sms' version)
- [ ] Update Host<->Device transfer exercise + slides with proper pinned mem approach
- [ ] Add zero-copy transfer to Host<->Device transfer exercise + slides
- [ ] Sort out Visual Studio solutions (try using VS 2010)
- [x] Add Unix Makefiles
- [ ] Instructions for using HP server and Zoo
- [ ] Add deps to part 1 slides
- [x] Remove native functions in Bilaterial exercise
- [ ] Update intro with correct path to all course material
- [x] Add links to IWOCL talks to the outro slides 
