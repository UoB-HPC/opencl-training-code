

ifndef CXX
	CXX = c++
endif

INC = ../../common
FLAGS = -O3
LDFLAGS = -lm -lOpenCL 

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

SRC = bilateral.cpp
EXE = bilateral

$(EXE): $(SRC)
	$(CXX) $(FLAGS) -I $(INC) $(SRC) $(LDFLAGS) -o $(EXE)

clean:
	rm -f $(EXE) output.bmp

