INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 #-lcudart

# NVCCFLAGS	:= -lineinfo -arch=sm_20 #--ptxas-options=-v --use_fast_math
NVCCFLAGS	:= -lineinfo -arch=sm_50 --std=c++11 #--ptxas-options=-v --use_fast_math

CFLAGS := -Wall -g
CXXFLAGS := -Wall -g
CXX := g++ --std=c++11

all: serial_batch serial_online sgd_cuda read_csv #thurst_simple sum_rows

serial_batch: serial_code_batch.c Makefile
	cc serial_code_batch.c -o serial_batch.o $(CFLAGS)
	
serial_online: serial_code_online.c Makefile
	cc serial_code_online.c -o serial_online.o $(CFLAGS)
	
sgd_cuda: cuda_gradient_descent.cu Makefile
	nvcc cuda_gradient_descent.cu -o sgd_cuda.o $(NVCCFLAGS) $(INC) $(LIB)

# Thrust examples
#thurst_simple: basic_vector_thrust.cu Makefile
#	nvcc basic_vector_thrust.cu -o thrust_simple $(NVCCFLAGS) $(INC) $(LIB)
	
#sum_rows: sum_rows.cu Makefile
#	nvcc sum_rows.cu -o sum_rows $(NVCCFLAGS) $(INC) $(LIB)
	
read_csv: read_csv.cc Makefile
	$(CXX) read_csv.cc -o read_csv.o $(CXXFLAGS)

clean:
	rm -f serial_batch.o serial_online.o sgd_cuda.o read_csv.o

