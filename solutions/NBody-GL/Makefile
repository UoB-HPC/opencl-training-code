#
# This code is released under the "attribution CC BY" creative commons license.
# In other words, you can use it in any way you see fit, including commercially,
# but please retain an attribution for the original authors:
# the High Performance Computing Group at the University of Bristol.
# Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
#

CC = cc
CXX = c++

INC = ../../common
CFLAGS = -std=c99 -O3
CXXFLAGS = -std=c++11 -O3
LIBS = -lm
LDFLAGS = -lOpenCL -lGL -lrt

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LDFLAGS = -lm -framework OpenCL -framework OpenGL -framework CoreGraphics
	ifeq ($(shell which sdl2-config > /dev/null; echo $$?), 0)
		LDFLAGS += $(shell sdl2-config --cflags --libs)
	else
		LDFLAGS += -framework SDL2
	endif
else
	LDFLAGS += $(shell sdl2-config --cflags --libs)
endif

EXES = nbody-c nbody-c++

all: $(EXES)

nbody-c: nbody.c
	$(CC) $(CFLAGS) -I $(INC) $< $(LIBS) $(LDFLAGS) -o $@

nbody-c++: nbody.cpp
	$(CXX) $(CXXFLAGS) -I $(INC) $< $(LIBS) $(LDFLAGS) -o $@

clean:
	rm -f $(EXES)
