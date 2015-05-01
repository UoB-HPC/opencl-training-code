

ifndef CXX
	CXX = c++
endif

INC = ../../common
FLAGS = -O3
LIBS = -lOpenCL

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL
endif

SRC = nbody.cpp
EXE = nbody

all:
	$(CXX) $(FLAGS) -I $(INC) $(SRC) $(LIBS) -o $(EXE)

clean:
	rm -f $(EXE)

