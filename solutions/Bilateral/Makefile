#
# This code is released under the "attribution CC BY" creative commons license.
# In other words, you can use it in any way you see fit, including commercially,
# but please retain an attribution for the original authors:
# the High Performance Computing Group at the University of Bristol.
# Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
#

CXX = c++

CXXFLAGS = -std=c++11 -O3 -I ../../common
LDFLAGS  = -lm -lOpenCL

PLATFORM = $(shell uname -s)

ifeq ($(PLATFORM), Darwin)
	LDFLAGS = -lm -framework OpenCL
	ifeq ($(shell which sdl2-config > /dev/null; echo $$?), 0)
		LDFLAGS += $(shell sdl2-config --cflags --libs)
	else
		LDFLAGS += -framework SDL2
	endif
else
	LDFLAGS += $(shell sdl2-config --cflags --libs)
endif

EXES = bilateral_meta bilateral_opt bilateral_images

all: $(EXES)

bilateral_meta: bilateral_meta.cpp ../../common/*.hpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

bilateral_opt: bilateral_opt.cpp ../../common/*.hpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

bilateral_images: bilateral_images.cpp ../../common/*.hpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f $(EXES) output.bmp

