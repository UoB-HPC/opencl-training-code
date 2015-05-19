#
# This code is released under the "attribution CC BY" creative commons license.
# In other words, you can use it in any way you see fit, including commercially,
# but please retain an attribution for the original authors:
# the High Performance Computing Group at the University of Bristol.
# Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
#

ifndef CXX
	CXX = c++
endif

INC = ../../common
FLAGS = -O3
LDFLAGS = -lm -lOpenCL -lGL

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

SRC = nbody.cpp
EXE = nbody

$(EXE): $(SRC)
	$(CXX) $(FLAGS) -I $(INC) $(SRC) $(LDFLAGS) -o $(EXE)

clean:
	rm -f $(EXE)

