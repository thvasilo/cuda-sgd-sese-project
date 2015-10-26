INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 #-lcudart

# NVCCFLAGS	:= -lineinfo -arch=sm_20 #--ptxas-options=-v --use_fast_math
NVCCFLAGS	:= -lineinfo -arch=sm_50 --std=c++11 #-g -G --ptxas-options=-v --use_fast_math

CFLAGS := -Wall -g
CXXFLAGS := -Wall -Wextra -g --std=c++11 -pedantic
CXX := g++

all:  sgd_thrust.o # read_csv_thrust.o

serial_batch.o: serial_code_batch.c Makefile
	cc serial_code_batch.c -o serial_batch.o $(CFLAGS)
	
serial_online.o: serial_code_online.c Makefile
	cc serial_code_online.c -o serial_online.o $(CFLAGS)
	
sgd_cuda.o: cuda_gradient_descent.cu Makefile
	nvcc cuda_gradient_descent.cu -o sgd_cuda.o $(NVCCFLAGS) $(INC) $(LIB)

# Thrust examples
#thurst_simple.o: basic_vector_thrust.cu Makefile
#	nvcc basic_vector_thrust.cu -o thrust_simple $(NVCCFLAGS) $(INC) $(LIB)
#	
#sum_rows.o: sum_rows.cu Makefile
#	nvcc sum_rows.cu -o sum_rows $(NVCCFLAGS) $(INC) $(LIB)
#
#thrust_vector_transform.o: thrust_vector_transform.cu Makefile
#	nvcc thrust_vector_transform.cu -o thrust_vector_transform.o $(NVCCFLAGS) $(INC) $(LIB)
#	
#zip.o: zip.cu Makefile
#	nvcc zip.cu -o zip.o $(NVCCFLAGS) $(INC) $(LIB)
#
#stride.o: stride.cu Makefile
#	nvcc stride.cu -o stride.o $(NVCCFLAGS) $(INC) $(LIB)

read_csv.o: read_csv.cc
	$(CXX) $(CXXFLAGS) read_csv.cc -c 

sgd_thrust.o: sgd_thrust.cu Makefile
	nvcc sgd_thrust.cu -o sgd_thrust.o $(NVCCFLAGS) $(INC) $(LIB)

read_csv_thrust.o: read_csv_thrust.cu Makefile
	nvcc read_csv_thrust.cu -o read_csv_thrust.o $(NVCCFLAGS) $(INC) $(LIB)

clean:
	rm -f serial_batch.o serial_online.o sgd_cuda.o read_csv.o sgd_thrust.o read_csv_thrust.o

