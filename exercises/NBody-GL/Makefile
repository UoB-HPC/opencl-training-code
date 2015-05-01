

ifndef CXX
	CXX = c++
endif

INC = ../../common
FLAGS = -O3
LIBS = -lm
LDFLAGS = -lOpenCL -lGL

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LDFLAGS = -framework OpenCL -framework OpenGL -framework CoreGraphics
endif

SDL = $(shell sdl2-config --cflags --libs)

SRC = nbody.cpp
EXE = nbody

all:
	$(CXX) $(FLAGS) -I $(INC) $(SRC) $(LIBS) $(LDFLAGS) $(SDL) -o $(EXE)

clean:
	rm -f $(EXE)

