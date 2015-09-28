INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 #-lcudart

# NVCCFLAGS	:= -lineinfo -arch=sm_20 #--ptxas-options=-v --use_fast_math
NVCCFLAGS	:= -lineinfo -arch=sm_50 --std=c++11 #-g -G --ptxas-options=-v --use_fast_math

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

#thrust_vector_transform: thrust_vector_transform.cu Makefile
#	nvcc thrust_vector_transform.cu -o thrust_vector_transform.o $(NVCCFLAGS) $(INC) $(LIB)
#	
#zip: zip.cu Makefile
#	nvcc zip.cu -o zip.o $(NVCCFLAGS) $(INC) $(LIB)
	
read_csv: read_csv.cc Makefile
	$(CXX) read_csv.cc -o read_csv.o $(CXXFLAGS)

stride: stride.cu Makefile
	nvcc stride.cu -o stride.o $(NVCCFLAGS) $(INC) $(LIB)

cuda_sgd: cuda_sgd.cu Makefile
	nvcc cuda_sgd.cu -o cuda_sgd.o $(NVCCFLAGS) $(INC) $(LIB)

clean:
	rm -f serial_batch.o serial_online.o sgd_cuda.o read_csv.o

