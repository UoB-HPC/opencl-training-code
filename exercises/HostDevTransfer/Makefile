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
LDFLAGS = -lOpenCL -lrt

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LDFLAGS = -framework OpenCL
endif

EXES = transfer-c transfer-c++

all: $(EXES)

transfer-c: transfer.c ../../common/*.h
	$(CC) $(CFLAGS) transfer.c $(LDFLAGS) -o $@

transfer-c++: transfer.cpp ../../common/*.hpp
	$(CXX) $(CXXFLAGS) transfer.cpp $(LDFLAGS) -o $@

.PHONY: clean
clean:
	rm -f $(EXES)

