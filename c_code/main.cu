#include "typedefs.cuh"
#include "sgd_io.cuh"
#include "sgd_thrust.cuh"
#include "sampling.cuh"

#include <algorithm>
#include <cmath>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>

using namespace thrust::placeholders;

// Function to divide tasks up to threads
// Arguments: a: number of items to divide, b: desired number of threads in each block
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

// Usage: Run with all arguments:
// args: [learning_rate] [iterations] [data_csv_file] [num_rows] [num_features] [batchsize]
// Setting batchsize to 0 uses the full data at each iteration
// NB: We are assuming that the csv has the format [features],[label]
// i.e. the last column is the label, and all others are features.
// num_features should equal the number of features only, i.e. the number of columns in the csv minus 1
// e.g.: > ./main 0.001 10 data/5xy.csv 40 1
// or with real data: > ./main 0.000004 500 data/ENB2012_data_Y1.csv 768 8
int main(int argc, char **argv) {

	if	(argc != 7) {
		std::cout << "usage: ./sgd_thrust.o [learning_rate] "
				"[iterations] [data_csv_file] [num_rows] [num_features] [batchsize]" << std::endl;
		return 1;
	}

	float learning_rate = atof(argv[1]);
	const int MAX_EPOCHS = atoi(argv[2]);
	const std::string filename = argv[3];
	const int R = atoi(argv[4]);
	const int C = atoi(argv[5]);
	const int batchsize = (atoi(argv[6])  == 0) ? R : atoi(argv[6]);
	const int num_batches = (int)std::floor(R/(float)batchsize);

	// std::cout << "lr: " << learning_rate << std::endl;
	// std::cout << "batchsize: " << batchsize << std::endl;
	// std::cout << "num_batches: " << num_batches << std::endl;

	cudaEvent_t start_memory;
	cudaEvent_t stop_memory;

	// Create the events
	cudaEventCreate(&start_memory);
	cudaEventCreate(&stop_memory);

	// Start recording
	cudaEventRecord(start_memory);
	// The number of threads we allocate per block
	const int THREADS_PER_BLOCK = batchsize;

	// Initialize data vector on host
	thrust_host_float data_h(R * C);

	// Initialize labels vector on host
	thrust_host_float labels_h(R);

	// Read data from csv file into host vectors
	read_csv(filename, data_h, labels_h, R, C);

	// Copy data from host vectors to device
	// note: d_vec.data() returns a device_ptr
	thrust_dev_float data_d = data_h;
	float * data_raw_ptr = thrust::raw_pointer_cast(data_d.data());
	thrust_dev_float labels_d = labels_h;
	float * labels_raw_ptr = thrust::raw_pointer_cast(labels_d.data());

	// Initialize weights
	thrust_dev_float weights(C);
	thrust::default_random_engine rng;
	thrust::uniform_real_distribution<float> weight_dist(0.0, 0.01);
	float * weights_raw_ptr = thrust::raw_pointer_cast(weights.data());
	for (size_t i = 0; i < weights.size(); i++) {
				weights[i] = weight_dist(rng);
	}

	// Initialize gradients
	thrust_dev_float gradients(C * batchsize);
	float * gradients_raw_ptr = thrust::raw_pointer_cast(gradients.data());

	//Initialize errors vector
	thrust_dev_float errors(R);
	float * errors_raw_ptr = thrust::raw_pointer_cast(errors.data());

	// Allocate storage for row sums and indices
	thrust_dev_float row_sums(C);
	thrust_dev_int row_indices(C);

	// Initialize batch indices vector
	thrust_dev_int batch_indices(R);
	int * batch_indices_ptr = thrust::raw_pointer_cast(batch_indices.data());
	// Fill indices vector, we first create and index vector, shuffle it and copy to device vector
	std::vector<int> ind_vector(R);
	for (int i = 0; i < R; ++i) {
		ind_vector[i] = i;
	}
	std::random_shuffle(ind_vector.begin(), ind_vector.end());
	batch_indices = ind_vector;
//	print_vector(batch_indices, "batch_indices");

	// Now measure the differences
	cudaEventRecord(stop_memory);
	cudaEventSynchronize(stop_memory);
	float miliseconds_memory = 0;
	cudaEventElapsedTime(&miliseconds_memory, start_memory, stop_memory);
	printf("Memory time = %f \n", miliseconds_memory);

	cudaEvent_t start;
	cudaEvent_t stop;

	// Create the events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// Start measuring the event
	cudaEventRecord(start);
	// Get the first time	
	// TODO: Put iterations in SGD_Step function
	for (int epoch = 1; epoch <= MAX_EPOCHS; ++epoch) {
		// We shuffle the data indexes before the start of each epoch
		// TODO: How is performance if we shuffle the data instead of indices, i.e. having sequential accesses instead of random?
		std::random_shuffle ( ind_vector.begin(), ind_vector.end());
		batch_indices = ind_vector; // TODO: Remove host-device copy, can we shuffle on the GPU instead?
		for (int batch = 0; batch < num_batches; ++batch) {
			// Reset gradients and errors
			thrust::fill(gradients.begin(), gradients.end(), 0.0);

			//Calculate the gradient vector for each datapoint
			calculate_gradients<<<iDivUp(batchsize, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(//iDivUp(batchsize, THREADS_PER_BLOCK)
					data_raw_ptr,
					labels_raw_ptr,
					weights_raw_ptr,
					batch_indices_ptr  + (batch * batchsize),
					gradients_raw_ptr,
					batchsize,
					C);

//			print_matrix(gradients, "weight_gradients", C, batchsize);

			// Sum/reduce the gradient vectors
			thrust::fill(row_sums.begin(), row_sums.end(), 0.0);
			thrust::fill(row_indices.begin(), row_indices.end(), 0.0);
			calculate_row_sums(C, batchsize, gradients, row_sums, row_indices);

//			print_vector(row_sums, "gradients");

			// Scale gradient sum vector
			thrust::for_each(row_sums.begin(), row_sums.end(), _1 / (float)batchsize);

			//Update the weight vector
			float a = -(learning_rate / std::pow(epoch, 0.25));
			//		std::cout << "a: " << a << std::endl;
			// Thrust SAXPY
			thrust::transform(row_sums.begin(), row_sums.end(),  // input range #1
					weights.begin(),           // input range #2
					weights.begin(),           // output range
					a * _1 + _2);        // placeholder expression
//			print_vector(weights, "weights");
		}
		if	(epoch % 100 == 0) {
			thrust::fill(errors.begin(), errors.end(), 0.0);
			// Calculate the squared error for each data point
			squared_errors<<<iDivUp(R, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
					data_raw_ptr,
					labels_raw_ptr,
					weights_raw_ptr,
					errors_raw_ptr,
					R,
					C);
			// Reduce/sum the errors
			float sq_err_sum = thrust::reduce(errors.begin(), errors.end());
			//printf("epoch: %d\n", epoch);
			// Print weights and squared error sum
			// print_vector(weights, "weights");
			// std::cout << "Squared error sum: " << sq_err_sum << std::endl;
			// print_matrix(gradients, "weight_gradients", R, C);
		}

	}


	// Print final weights and squared error sum
	// Calculate the squared error for each data point
	//		cudaDeviceSynchronize();
	squared_errors<<<iDivUp(R, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
			data_raw_ptr,
			labels_raw_ptr,
			weights_raw_ptr,
			errors_raw_ptr,
			R,
			C);

	//	float sq_err_sum = thrust::reduce(errors.begin(), errors.end());
	//	print_vector(weights, "final_weights");
	//	std::cout << "Final squared error sum: " << sq_err_sum << std::endl;

	// Print final quantities
	float sq_err_sum = thrust::reduce(errors.begin(), errors.end());
	std::cout << "Squared error sum: " << sq_err_sum << std::endl;
	print_vector(weights, "weights");	
	
	// Get the second time
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);
	printf("kernel time = %f \n", miliseconds);

	return 0;
}
