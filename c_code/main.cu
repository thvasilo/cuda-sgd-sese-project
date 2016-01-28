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
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace thrust::placeholders;

// Function to divide tasks up to threads
// Arguments: a: number of items to divide, b: desired number of threads in each block
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

void test_col_sums() {
	const int R = 5;
	const int C = 4;

	// Initialize data vector on host
	thrust_host_float data_h(R * C);

	// Initialize labels vector on host
	thrust_host_float labels(R);

	std::string filename = "data/col_sum_test.csv";
	// Read data from csv file into host vectors
	read_csv(filename, data_h, labels, R, C);

	print_matrix(data_h, "data_h", R, C);

	// Copy data from host vectors to device
	thrust_dev_float data_d = data_h;
	float* data_dev_ptr = thrust::raw_pointer_cast(data_d.data());

	// Initialize the column sum vector
	thrust_dev_float col_sums(C);

	calculate_column_sums(
			data_dev_ptr,
			col_sums,
			R,
			C);

	thrust_host_float col_sums_h = col_sums;
	print_vector(col_sums_h, "column_sums");
}

void test_matrix_scale() {
	const int R = 5;
	const int C = 4;

	// Initialize data vector on host
	thrust_host_float data_h(R * C);

	// Initialize labels vector on host
	thrust_host_float scaling_vector(R);

	std::string filename = "data/matrix_scale_test";
	// Read data from csv file into host vectors
	read_csv(filename, data_h, scaling_vector, R, C);

	print_matrix(data_h, "data_h", R, C);
	print_vector(scaling_vector, "scaling_vector");

	// Copy data from host vectors to device
	thrust_dev_float data_d = data_h;
	thrust::device_ptr<float> data_dev_ptr = data_d.data();
	thrust_dev_float scaling_d = scaling_vector;

	// Initialize the results matrix
	thrust_dev_float scaled_matrix(R*C);

	scale_matrix_rows_by_vector(
		data_dev_ptr,
		scaling_d,
		scaled_matrix,
		R,
		C);

	print_matrix(scaled_matrix, "data_h_scaled", R, C);
}

void test_gemv() {

	const int R = 5;
	const int C = 4;
	const int batchsize = 5;

	// Initialize data vector on host
	thrust_host_float data_h(R * C);

	// Initialize labels vector on host
	thrust_host_float labels_h(R);

	std::string filename = "data/gemv_test";
	// Read data from csv file into host vectors
	read_csv(filename, data_h, labels_h, R, C);

	print_matrix(data_h, "data_h", R, C);


	// Copy data from host vectors to device
	// The data matrix has all elements equal to their row index + 1
	thrust_dev_float data_d = data_h;
	const float * data_raw_ptr = thrust::raw_pointer_cast(data_d.data());
	// All the labels are 4.0
	thrust_dev_float labels_d = labels_h;
	const float * labels_raw_ptr = thrust::raw_pointer_cast(labels_d.data());

	// Initialize weights to 1
	thrust_dev_float weights(C);
	thrust::fill(weights.begin(), weights.end(), 1.0);
	float * weights_raw_ptr = thrust::raw_pointer_cast(weights.data());

	print_vector(weights, "weights");
	print_vector(labels_h, "labels_h");
	// Initialize loss derivative vector
	thrust_dev_float loss_derivative(batchsize);
	float * loss_derivative_raw_ptr = thrust::raw_pointer_cast(loss_derivative.data());

	calculate_loss_derivative_cublas(
		data_raw_ptr,
		labels_raw_ptr,
		weights_raw_ptr,
		loss_derivative_raw_ptr,
		R,
		C,
		batchsize);

	thrust_host_float loss_derivative_host(batchsize);
	loss_derivative_host = loss_derivative;
	print_vector(loss_derivative_host, "loss_derivative");
}

// Usage: Run with all arguments:
// args: [learning_rate] [iterations] [data_csv_file] [num_rows] [num_features] [batchsize]
// Setting batchsize to 0 uses the full data at each iteration
// NB: We are assuming that the csv has the format [features],[label]
// i.e. the last column is the label, and all others are features.
// num_features should equal the number of features only, i.e. the number of columns in the csv minus 1
// e.g.: > ./main 0.00001 10 data/5xy.csv 40 1 0
int main(int argc, char **argv) {

//	test_gemv();
//
//	test_matrix_scale();
//
//	test_col_sums();
//
//	return 0;

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
	thrust_dev_float gradients(batchsize * C);
	float * gradients_raw_ptr = thrust::raw_pointer_cast(gradients.data());

	// Initialize loss derivative vector
	thrust_dev_float loss_derivative(batchsize);
	float * loss_derivative_raw_ptr = thrust::raw_pointer_cast(loss_derivative.data());

	//Initialize errors vector
	thrust_dev_float errors(R);
	float * errors_raw_ptr = thrust::raw_pointer_cast(errors.data());

	// Allocate storage for row sums and indices
	thrust_dev_float col_sums(C);

	// Initialize batch indices vector
	thrust_dev_int batch_indices(R);
	int * batch_indices_ptr = thrust::raw_pointer_cast(batch_indices.data());
	// Fill indices vector, we first create and index vector, shuffle it and copy to device vector
	std::vector<int> ind_vector(R);
	for (int i = 0; i < R; ++i) {
		ind_vector[i] = i;
	}
	// Shuffle the vector on the host, and copy to the device
	std::random_shuffle(ind_vector.begin(), ind_vector.end());
	batch_indices = ind_vector;

	// Now measure the differences
	cudaEventRecord(stop_memory);
	cudaEventSynchronize(stop_memory);
	float miliseconds_memory = 0;
	cudaEventElapsedTime(&miliseconds_memory, start_memory, stop_memory);
	printf("Memory time = %f ms\n", miliseconds_memory);

	cudaEventDestroy(start_memory);
	cudaEventDestroy(stop_memory);

	cudaEvent_t start;
	cudaEvent_t stop;

	// Create the events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// Start measuring the event
	cudaEventRecord(start);

	for (int epoch = 1; epoch <= MAX_EPOCHS; ++epoch) {
		// We shuffle the data indexes before the start of each epoch
		// TODO: How is performance if we shuffle the data instead of indices, i.e. having sequential accesses instead of random?
		std::random_shuffle ( ind_vector.begin(), ind_vector.end());
		batch_indices = ind_vector; // TODO: Remove host-device copy, can we shuffle on the GPU instead?
		for (int batch = 0; batch < num_batches; ++batch) {
			// Reset gradients and errors
			thrust::fill(gradients.begin(), gradients.end(), 0.0);
			//thrust::fill(loss_derivative.begin(), loss_derivative.end(), 0.0);

			// Calculate the loss derivative vector
			// TODO: We are assuming that data_array_d points to the first element in the batch in the data array,
			// and label_vector_d points to the corresponding element in the label vector
			float * cur_batch_data_ptr = data_raw_ptr;
			thrust::device_ptr<float> cur_batch_data_dev_ptr(cur_batch_data_ptr);
			float * cur_batch_labels_ptr = labels_raw_ptr;

			calculate_loss_derivative_cublas(
					cur_batch_data_ptr, // TODO: Fix offsets to be consistent with current batch
					cur_batch_labels_ptr,
					weights_raw_ptr,
					loss_derivative_raw_ptr,
					R,
					C,
					batchsize);

			// The gradient matrix is equal to the feature matrix of the batch scaled by the loss derivative vector
			scale_matrix_rows_by_vector(
				cur_batch_data_dev_ptr,
				loss_derivative,
				gradients, // Result stored in gradient matrix (batchsize*C)
				batchsize,
				C);

			// Once we have the scaled data matrix, i.e. the gradients we need to sum the columns and scale to get the avg. gradient vector.
			calculate_column_sums(
				cur_batch_data_ptr,
				col_sums,
				batchsize,
				C);

			// Scale gradient sum vector to obtain avg. gradient vector
			thrust::for_each(col_sums.begin(), col_sums.end(), _1 / (float)batchsize);

			//Update the weight vector
			float a = -(learning_rate / std::pow(epoch, 0.25));

			// Thrust SAXPY, used to update the weights vector
			thrust::transform(col_sums.begin(), col_sums.end(),  // input range #1
					weights.begin(),           // input range #2
					weights.begin(),           // output range
					a * _1 + _2);        // placeholder expression
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
		}

	}


	// Print final weights and squared error sum
	// Calculate the squared error for each data point
	squared_errors<<<iDivUp(R, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
			data_raw_ptr,
			labels_raw_ptr,
			weights_raw_ptr,
			errors_raw_ptr,
			R,
			C);

	// Print final quantities
	float sq_err_sum = thrust::reduce(errors.begin(), errors.end());
	std::cout << "Squared error sum: " << sq_err_sum << std::endl;
	print_vector(weights, "weights");	
	
	// Get the second time
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);
	printf("kernel time = %f ms\n", miliseconds);


	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
