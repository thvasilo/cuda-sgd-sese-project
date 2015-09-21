INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

# NVCCFLAGS	:= -lineinfo -arch=sm_20 #--ptxas-options=-v --use_fast_math
NVCCFLAGS	:= -lineinfo -arch=sm_50 #--ptxas-options=-v --use_fast_math

CFLAGS := -Wall -g

all: serial_batch serial_online sgd_cuda

serial_batch: serial_code_batch.c Makefile
	cc serial_code_batch.c -o serial_batch $(CFLAGS)
	
serial_online: serial_code_online.c Makefile
	cc serial_code_online.c -o serial_online $(CFLAGS)
	
sgd_cuda: cuda_gradient_descent.cu Makefile
	nvcc cuda_gradient_descent.cu -o sgd_cuda $(NVCCFLAGS) $(INC) $(LIB)

clean:
	rm -f serial_batch serial_online sgd_cuda

