

ifndef CXX
	CXX = c++
endif

INC = ../../common
FLAGS = -O3
LIBS = -lm
OCL_LIB = -lOpenCL 

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	OCL_LIB = -framework OpenCL
endif

SDL = $(shell sdl2-config --cflags --libs)

SRC = bilateral.cpp
EXE = bilateral

all:
	$(CXX) $(FLAGS) -I $(INC) $(SRC) $(LIBS) $(OCL_LIB) $(SDL) -o $(EXE)

clean:
	rm -f $(EXE) output.bmp

