CUDA_HOME := /usr/local/cuda-7.5
INC	:= -I /json#-I $(CUDA_HOME)/include -I.
# Need to use lculibos when using lcublas_static, culbas_device necessary for cuBLAS calls within kernel
LIB	:= -L $(CUDA_HOME)/lib64 -lcudadevrt -lcublas_static -lculibos -lcublas_device #-L json/lib -ljsoncpp# -lcudart_static  -lcublas_static 

# NB! arch=sm_35 is the lowest supported for dynamic parallelism
# NVCCFLAGS	:= -lineinfo -arch=sm_20 #--ptxas-options=-v --use_fast_math
NVCCFLAGS := -lineinfo -arch=sm_50 --std=c++11#-g -G --ptxas-options=-v --use_fast_math
#NVCCFLAGS	:= -lineinfo -arch=sm_35 --std=c++11 #-g -G --ptxas-options=-v --use_fast_math

OBJECTS := jsoncpp.o sgd_io.o sgd_thrust.o testing.o main.o 

CFLAGS := -Wall -g
CXXFLAGS := -Wall -Wextra -g --std=c++11 -pedantic
CXX := g++
NVCC := nvcc

all:  $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) $(INC) $(LIB) -o main

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INC) -x cu -I. -dc $< -o $@
	
%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INC) -x cu -I. -dc $< -o $@

clean:
	rm -f *.o main
