

ifndef CXX
	CXX = c++
endif

INC = ../../common
FLAGS = -O3
LDFLAGS = -lm -lOpenCL 

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LDFLAGS = -lm -framework OpenCL
endif

SDL = $(shell sdl2-config --cflags --libs)

SRC = bilateral.cpp
EXE = bilateral

all:
	$(CXX) $(FLAGS) -I $(INC) $(SRC) $(LDFLAGS) $(SDL) -o $(EXE)

clean:
	rm -f $(EXE) output.bmp

