

ifndef CXX
	CXX = c++
endif

INC = ../../common
FLAGS = -O3
LDFLAGS = -lOpenCL

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LDFLAGS = -framework OpenCL
endif

SRC = deviceinfo.cpp
EXE = deviceinfo

$(EXE): $(SRC)
	$(CXX) $(FLAGS) -I $(INC) $(SRC) $(LDFLAGS) -o $(EXE)

clean:
	rm -f $(EXE)

