/*
This program calculates whole gradient descent (non-batch or batch equal to sample size)
using each thread to calculate one error. For comparative purposes.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>

__host__ __device__
float linear_model(float x, float w0, float w1){
    return w0 + w1 * x;
}

__host__  __device__
void evaluate_batch(float *y, float *x, int start,
					 int end, float w0, float w1){
  int i;
  
  for(i=start; i<end; i++){
    y[i] = linear_model(x[i], w0, w1);
  }
}

__host__
void print_vector(float *y, int N){
  int i;
  for(i=0; i<N; i++){
    printf("for i = %d the value of y = %f \n", i, y[i]);
  }
  printf("---------- \n");
}

__global__
void kernel(float *x, float *y, float w0_est, float w1_est, 
	    float gamma, float *grad0, float *grad1, int batch_size){

    int index = threadIdx.x + blockDim.x * blockIdx.x; 
    float x_thread = x[index];
    float y_predic = linear_model(x_thread, w0_est, w1_est);
    float error = y_predic - y[index];

    grad0[index] = error;
    grad1[index] = error * x[index];
}


int main(){
    // System parameters
    int i;
    int j;
    bool print = false;
    
    // First the main parameters
    int Ndata = 1000;
    int Nbatch = 10;
    int batch_size = Ndata / Nbatch;
    float w0 = 3;
    float w1 = 1;

    // Now the examples set
    float x[Ndata];
    float y[Ndata];
    float y_predic[Ndata];

    // Vector for the errors
    float grad0[Ndata];
    float grad1[Ndata];

    // Pointers in the device
    float *x_device;
    float *y_device;
    float *grad0_device;
    float *grad1_device;

    // Size of the memories
    int data_size = Ndata * sizeof(float);
    int grad_size = Ndata * sizeof(float);

    // Timer
    struct timeval t1, t2;
    double elapsedTime;

    // Allocate memories
    cudaMalloc((void**) &x_device, data_size);
    cudaMalloc((void**) &y_device, data_size);
    cudaMalloc((void**) &grad0_device, grad_size);
    cudaMalloc((void**) &grad1_device, grad_size);

    // We get x
    for(i=0; i<Ndata; i++){
    	x[i] = (float) rand() / (RAND_MAX);
    }
    // We get y
    evaluate_batch(y, x, 0, Ndata, w0, w1);

    // Copy data to the device
    cudaMemcpy(x_device, x, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y, data_size, cudaMemcpyHostToDevice);
    
    // Now the algorithm
    int Niterations = 10000;
    float gamma = 0.1;
    float w0_est = 0.5;
    float w1_est = 0.5;

    // Kernel variables
    dim3 dimGrid(Nbatch, 1, 1);
    dim3 dimBlock(batch_size, 1, 1);

    printf("Number of blocks = %d", Nbatch);
    printf("Number of threads per block = %d", batch_size);

    float error0;
    float error1;

    // Get the first time
    gettimeofday(&t1, NULL);

    for(i=0; i<Niterations; i++){
		// Calculate the errors
		kernel<<<dimGrid, dimBlock>>>(x_device, y_device, w0_est, w1_est,
						  gamma, grad0_device, grad1_device, batch_size);
		// Copy the errors to the host
		cudaMemcpy(grad0, grad0_device, grad_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(grad1, grad1_device, grad_size, cudaMemcpyDeviceToHost);
		// print_vector(grad0, Ndata);
	
		// Add the errors
		error0 = 0;
		error1 = 0;

		for(j=1; j < Ndata; j++){
			error0 += grad0[j];
			error1 += grad1[j];
			}

		// Updates
		w0_est -= gamma * error0 / Ndata;
		w1_est -= gamma * error1 / Ndata;
		// printf("The estimated values for w are %f %f \n", w0_est, w1_est);

    }
    cudaDeviceSynchronize();
    // Get the second time
    gettimeofday(&t2, NULL);

    // Print the real and the estimated values
    printf("The real values for w are %f %f \n", w0, w1);
    printf("The estimated values for w are %f %f \n", w0_est, w1_est);

    // Free the memory
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(grad0_device);
    cudaFree(grad1_device);

    // print data and the estimation
    if(print){
		print_vector(y, Ndata);
		print_vector(x, Ndata);
		print_vector(y_predic, Ndata);
    }

    // Computer and print the elapsed time in miliseconds
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;  // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    printf("Elapsed Time = %f ms \n", elapsedTime);
	
    
    return 0;

}
