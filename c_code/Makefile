INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 #-lcudart

# NVCCFLAGS	:= -lineinfo -arch=sm_20 #--ptxas-options=-v --use_fast_math
NVCCFLAGS	:= -lineinfo -arch=sm_50 --std=c++11 #-g -G --ptxas-options=-v --use_fast_math

OBJECTS := sgd_io.o sampling.o sgd_thrust.o main.o

CFLAGS := -Wall -g
CXXFLAGS := -Wall -Wextra -g --std=c++11 -pedantic
CXX := g++

all:  $(OBJECTS)
	nvcc $(NVCCFLAGS) $(OBJECTS) -o main

%.o: %.cu #%.cpp
	nvcc $(NVCCFLAGS) -x cu -I. -dc $< -o $@

clean:
	rm -f *.o main