#
# This code is released under the "attribution CC BY" creative commons license.
# In other words, you can use it in any way you see fit, including commercially,
# but please retain an attribution for the original authors:
# the High Performance Computing Group at the University of Bristol.
# Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
#

CC = cc
CXX = c++

CFLAGS = -std=c99 -O3 -I ../../common
CXXFLAGS = -std=c++11 -O3 -I ../../common
LDFLAGS  = -lm -lOpenCL -lrt
SDLFLAGS = -D USE_SDL

PLATFORM = $(shell uname -s)

ifeq ($(PLATFORM), Darwin)
	LDFLAGS = -lm -framework OpenCL
	ifeq ($(shell which sdl2-config > /dev/null; echo $$?), 0)
		SDLFLAGS += $(shell sdl2-config --cflags --libs)
	else
		SDLFLAGS += -framework SDL2
	endif
else
	SDLFLAGS += $(shell sdl2-config --cflags --libs)
endif

EXES = bilateral-c bilateral-c++

all: $(EXES)

bilateral-c: bilateral.c ../../common/*.h
	$(CC) $(CFLAGS) bilateral.c $(LDFLAGS) -o $@

bilateral-c++: bilateral.cpp ../../common/*.hpp
	$(CXX) $(CXXFLAGS) bilateral.cpp $(LDFLAGS) -o $@

use-sdl: CFLAGS += $(SDLFLAGS) CXXFLAGS += $(SDLFLAGS)
use-sdl: all

clean:
	rm -f $(EXES) output.bmp
