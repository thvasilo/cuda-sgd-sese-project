#include "typedefs.cuh"
#include "sgd_io.cuh"
#include "sgd_thrust.cuh"
#include "sampling.cuh"
#include "testing.cuh"
#include "json/json.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <assert.h>

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
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

struct CudaTimings {
	float shuffle_time = 0.0;
	float grad_mat_time = 0.0;
	float gradient_scale_time = 0.0;
	float row_sum_time = 0.0;
	float saxpy_time = 0.0;
	float total_gradient_time = 0.0;
};

struct CuBLASTimings {
	float shuffle_time = 0.0;
	float derivative_time = 0.0;
	float matrix_scale_time = 0.0;
	float column_sum_time = 0.0;
	float saxpy_time = 0.0;
	float total_gradient_time = 0.0;
};

// TODO: Take the iterations to separate source file?

/**
 * One iteration (epoch) of SGD using plain CUDA calls
 */
void cuda_iteration(
	CudaTimings* timings,
	std::vector<int>& ind_vector,
	thrust_dev_float& gradients,
	thrust_dev_float& weights,
	thrust_dev_float& row_sums,
	thrust_dev_int& row_indices,
	thrust_dev_int& batch_indices,
	float* data_raw_ptr, // TOTEST: If creating raw ptrs is lightweight, we can do it at every iteration as well, reduce no. of args.
	float* labels_raw_ptr,
	float* weights_raw_ptr,
	float* gradients_raw_ptr,
	int* batch_indices_ptr,
	const int THREADS_PER_BLOCK,
	float learning_rate,
	int epoch,
	int num_batches,
	int batchsize,
	int R,
	int C
	) {

	float shuffle_time = 0.0;
	float grad_mat_time = 0.0;
	float gradient_scale_time = 0.0;
	float row_sum_time = 0.0;
	float saxpy_time = 0.0;
	float total_gradient_time = 0.0;

	// Create the events and start recording
	cudaEvent_t start_shuffle, stop_shuffle;
	create_events_and_start(start_shuffle, stop_shuffle);

	// We shuffle the data indexes before the start of each epoch
	std::random_shuffle ( ind_vector.begin(), ind_vector.end());
	batch_indices = ind_vector; // TODO: Remove host-device copy, can we shuffle on the GPU instead?

	measure_event(start_shuffle, stop_shuffle, shuffle_time, "shuffle_time");
	for (int batch = 0; batch < num_batches; ++batch) {
		// Start recording the gradient time
		cudaEvent_t start_total_gradient, stop_total_gradient;
		create_events_and_start(start_total_gradient, stop_total_gradient);

		// Reset gradients and errors
		thrust::fill(gradients.begin(), gradients.end(), 0.0);

		// Start recording the gradient matrix creation time
		cudaEvent_t start_grad_mat, stop_grad_mat;
		create_events_and_start(start_grad_mat, stop_grad_mat);

		//Calculate the gradient vector for each datapoint
		calculate_gradients<<<iDivUp(batchsize, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
				data_raw_ptr,
				labels_raw_ptr,
				weights_raw_ptr,
				batch_indices_ptr  + (batch * batchsize),
				gradients_raw_ptr,
				batchsize,
				C);

		measure_event(start_grad_mat, stop_grad_mat, grad_mat_time, "gradient_matrix_time");

		cudaEvent_t start_row_sum, stop_row_sum;
		create_events_and_start(start_row_sum, stop_row_sum);

		// Sum/reduce the gradient vectors
		thrust::fill(row_sums.begin(), row_sums.end(), 0.0);
		thrust::fill(row_indices.begin(), row_indices.end(), 0.0);
		calculate_row_sums(C, batchsize, gradients, row_sums, row_indices);

		measure_event(start_row_sum, stop_row_sum, row_sum_time, "col_sum_time");

		// Create the events and start recording
		cudaEvent_t start_scale, stop_scale;
		create_events_and_start(start_scale, stop_scale);

		// Scale gradient sum vector
		thrust::transform(row_sums.begin(), row_sums.end(), row_sums.begin(), scale_functor(batchsize));

		measure_event(start_scale, stop_scale, gradient_scale_time, "gradient_scale_time");

		measure_event(start_total_gradient, stop_total_gradient, total_gradient_time, "total_gradient_time");

		// Create the events and start recording
		cudaEvent_t start_saxpy, stop_saxpy;
		create_events_and_start(start_saxpy, stop_saxpy);

		//Update the weight vector
		float a = -(learning_rate / std::pow(epoch, 0.25));

		// Thrust SAXPY
		thrust::transform(row_sums.begin(), row_sums.end(),  // input range #1
				weights.begin(),           // input range #2
				weights.begin(),           // output range
				a * _1 + _2);        // placeholder expression
		measure_event(start_saxpy, stop_saxpy, saxpy_time, "saxpy_time");

		// Update timings object
		timings->shuffle_time += shuffle_time;
		timings->grad_mat_time += grad_mat_time;
		timings->gradient_scale_time += gradient_scale_time;
		timings->row_sum_time += row_sum_time;
		timings->saxpy_time += saxpy_time;
		timings->total_gradient_time += total_gradient_time;
	}


}

/**
 * One iteration (epoch) of SGD using cuBLAS calls.
 */
void cublas_iteration(
		CuBLASTimings* timings,
		thrust_dev_float& data_d,
		thrust_dev_float& labels_d,
		thrust_dev_float& data_shuffled_d,
		thrust_dev_float& labels_shuffled_d,
		thrust_dev_float& gradients,
		thrust_dev_float& weights,
		thrust_dev_float& loss_derivative,
		thrust_dev_float& col_sums,
		std::vector<int>& ind_vector,
		thrust::device_vector<int> batch_indices_d,
		float* data_raw_ptr, // TOTEST: If creating raw ptrs is lightweight, we can do it at every iteration as well, reduce no. of args.
		float* labels_raw_ptr,
		float* weights_raw_ptr,
		float* loss_derivative_raw_ptr,
		float* gradients_raw_ptr,
		float learning_rate,
		int epoch,
		int num_batches,
		int batchsize,
		int R,
		int C
		) {
	float * data_shuffled_raw_ptr = thrust::raw_pointer_cast(data_shuffled_d.data());
	float * labels_shuffled_raw_ptr = thrust::raw_pointer_cast(labels_shuffled_d.data());
	// Initialize the timings
	float shuffle_time = 0.0;
	float derivative_time = 0.0;
	float matrix_scale_time = 0.0;
	float column_sum_time = 0.0;
	float saxpy_time = 0.0;
	float total_gradient_time = 0.0;

	// Create the events and start recording
	cudaEvent_t start_shuffle, stop_shuffle;
	create_events_and_start(start_shuffle, stop_shuffle);

	// We shuffle the data indexes before the start of each epoch on the host, and copy to the GPU
	std::random_shuffle ( ind_vector.begin(), ind_vector.end());
	// Currently we are shuffling the indices vector on host and copying to device.
	// Maybe it is possible to create a random permutation vector on the GPU, or shuffle an existing one (i.e. copy from device
	// only once.)
	batch_indices_d = ind_vector;

	// This creates a permutation of the data and copies it to data_shuffled_d and the same with labels.
	// TODO: Is there a way to avoid having the dev-dev copy at each iteration?
	permute_data_and_labels(
					data_d,
					labels_d,
					batch_indices_d,
					data_shuffled_d,
					labels_shuffled_d,
					R,
					C);

	// Record the elapsed time
	measure_event(start_shuffle, stop_shuffle, shuffle_time, "shuffle_time");

	for (int batch = 0; batch < num_batches; ++batch) {
		// Create the events and start recording
		cudaEvent_t start_total_gradient, stop_total_gradient;
		create_events_and_start(start_total_gradient, stop_total_gradient);

		// Reset gradients and errors
		thrust::fill(gradients.begin(), gradients.end(), 0.0); // TODO: Necessary?

		// Pointer offsets to be consistent with current batch
		int offset = batch * batchsize;
		float * cur_batch_data_ptr = data_shuffled_raw_ptr + (offset*C);
		thrust::device_ptr<float> cur_batch_data_dev_ptr(cur_batch_data_ptr);
		float * cur_batch_labels_ptr = labels_shuffled_raw_ptr + offset;

		// Calculate the loss derivative vector
		cudaEvent_t start_deriv, stop_deriv;
		create_events_and_start(start_deriv, stop_deriv);
		calculate_loss_derivative_cublas(
				cur_batch_data_ptr,
				cur_batch_labels_ptr,
				weights_raw_ptr,
				loss_derivative_raw_ptr,
				R,
				C,
				batchsize);
		measure_event(start_deriv, stop_deriv, derivative_time, "deriv_time");

		// The gradient matrix is equal to the feature matrix of the batch scaled row/element-wise by the loss derivative vector.
		cudaEvent_t start_scale, stop_scale;
		create_events_and_start(start_scale, stop_scale);
		scale_matrix_rows_by_vector(
			cur_batch_data_dev_ptr,
			loss_derivative,
			gradients, // Result stored in gradient matrix of size batchsize*C
			batchsize,
			C);
		measure_event(start_scale, stop_scale, matrix_scale_time, "matrix_scale_time");

		// Once we have the scaled data matrix, i.e. the gradients we need to sum the columns and scale to get
		// the avg. gradient vector.
		cudaEvent_t start_col_sum, stop_col_sum;
		create_events_and_start(start_col_sum, stop_col_sum);
		calculate_scaled_col_sums(
			gradients_raw_ptr,
			col_sums, // col_sums will now contain the sum of the columns in the gradient matrix, scaled by 1/batchsize
			batchsize,
			C,
			float(batchsize));
		// TODO: DAFAQ: Why does it still converge (and much faster) when I don't scale by the batch size?
		measure_event(start_col_sum, stop_col_sum, column_sum_time, "col_sum_time");

		// Gradient calculation finished
		measure_event(start_total_gradient, stop_total_gradient, total_gradient_time, "total_gradient_time");

		//Update the weight vector
		float a = -(learning_rate / std::pow(epoch, 0.25));

		// Thrust SAXPY, used to update the weights vector
		cudaEvent_t start_saxpy, stop_saxpy;
		create_events_and_start(start_saxpy, stop_saxpy);
		thrust::transform(col_sums.begin(), col_sums.end(),  // input range #1
				weights.begin(),           // input range #2
				weights.begin(),           // output range
				a * _1 + _2);        // placeholder expression
		measure_event(start_saxpy, stop_saxpy, saxpy_time, "saxpy_time");

		// Update the timings object
		timings->shuffle_time += shuffle_time;
		timings->derivative_time += derivative_time;
		timings->matrix_scale_time += matrix_scale_time;
		timings->column_sum_time += column_sum_time;
		timings->saxpy_time += saxpy_time;
		timings->total_gradient_time += total_gradient_time;
	}
}

/** Usage: Run with all arguments: [learning_rate] [iterations] [data_csv_file] [num_rows] [num_features] [batchsize] [cudaRun]
 *
 * Setting batchsize to 0 uses the full data at each iteration.
 * The argument [cudaRun] determines if we should use the plain CUDA or the cuBLAS code for the iterations.
 * Set to 1 for a plain CUDA run, 0 for a cuBLAS run.
 * NB: We are assuming that the csv has the format [features],[label]
 * i.e. the last column is the label, and all others are features.
 * num_features should equal the number of features only, i.e. the number of columns in the csv minus 1
 * e.g.: > ./main 0.00001 10 data/5xy.csv 40 1 0 0
 * will do a run with learning rate 0.00001 for 10 epochs, on the dataset data/5xy.csv, which has 40 rows and 1 feature column,
 * using the complete data as a single batch and use the cuBLAS codepath for the iterations.
**/
int main(int argc, char **argv) {

//	test_permutation();

//	test_gemv();
//
//	test_matrix_scale();
//
//	test_col_sums();
//	test_col_sum_and_scale();
//
	test_abs_error();

//	check_transfer_speed();

//	return 0;

	if	(argc != 8) {
		std::cout << "usage: ./sgd_thrust.o [learning_rate] "
				"[iterations] [data_csv_file] [num_rows] [num_features] [batchsize] [cudaRun]" << std::endl;
		return 1;
	}

	float learning_rate = atof(argv[1]);
	const int MAX_EPOCHS = atoi(argv[2]);
	const std::string filepath = argv[3];
	const int R = atoi(argv[4]);
	const int C = atoi(argv[5]);
	const int batchsize = (atoi(argv[6])  == 0) ? R : atoi(argv[6]);
	const int num_batches = (int)std::floor(R/(float)batchsize);
	const bool cudaRun = (atoi(argv[7])  == 1) ? true : false;
	// The number of threads we allocate per block
	const int THREADS_PER_BLOCK = 256;


	cudaEvent_t start_memory;
	cudaEvent_t stop_memory;
	create_events_and_start(start_memory, stop_memory);

	// Initialize data vector on host
	thrust_host_float data_h(R * C);
	std::cout << "Data size: ~" << float(R)*C*sizeof(float)/(1024*1024) << "MB" << std::endl;

	// Initialize labels vector on host
	thrust_host_float labels_h(R);

	cudaEvent_t start_csv;
	cudaEvent_t stop_csv;
	create_events_and_start(start_csv, stop_csv);
	// Read data from csv file into host vectors
	bool read_success = read_csv(filepath, data_h, labels_h, R, C);
	if (!read_success)
	{
		std::cout << "Could not read file, exiting..." << std::endl;
		return 0;
	}
	float csv_time = 0.0;
	measure_event(start_csv, stop_csv, csv_time, "read_csv_time");
	printf("%s time = %f ms\n", "csv read", csv_time);

	// Copy data from host vectors to device
	// note: d_vec.data() returns a device_ptr
	float copy_time = 0.0;
	cudaEvent_t start_copy;
	cudaEvent_t stop_copy;
	create_events_and_start(start_copy, stop_copy);
	thrust_dev_float data_d = data_h;
	float * data_raw_ptr = thrust::raw_pointer_cast(data_d.data());
	thrust_dev_float labels_d = labels_h;
	float * labels_raw_ptr = thrust::raw_pointer_cast(labels_d.data());
	measure_event(start_copy, stop_copy, copy_time, "gpu_copy_time");
	printf("%s time = %f ms\n", "GPU copy", copy_time);

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

	// Allocate storage for column sums and indices
	thrust_dev_float col_sums(C);

	// Allocate storage for matrix and vector shuffled copies.
	thrust_dev_float data_shuffled_d(R*C);
	float * data_shuffled_raw_ptr = thrust::raw_pointer_cast(data_shuffled_d.data());
	thrust_dev_float labels_shuffled_d(R);
	float * labels_shuffled_raw_ptr = thrust::raw_pointer_cast(labels_shuffled_d.data());

	// Initialize batch indices vector
	thrust_dev_int batch_indices_d(R);
	// Fill indices vector, we first create and index vector, shuffle it and copy to device vector
	std::vector<int> ind_vector(R);
	for (int i = 0; i < R; ++i) {
		ind_vector[i] = i;
	}
	// Copy the indices vector to the device
	batch_indices_d = ind_vector;

	// Initialize batch indices vector
	int * batch_indices_ptr = thrust::raw_pointer_cast(batch_indices_d.data());

	// Allocate storage for row sums and indices (used for plain CUDA iterations)
	thrust_dev_float row_sums(C);
	thrust_dev_int row_indices(C);

	// Record the complete data transfer time (read+transfer)
	cudaEventRecord(stop_memory);
	cudaEventSynchronize(stop_memory);
	float transfer_time = 0;
	cudaEventElapsedTime(&transfer_time, start_memory, stop_memory);
	cudaEventDestroy(start_memory);
	cudaEventDestroy(stop_memory);

	// Record the GPU time
	cudaEvent_t start, stop;
	create_events_and_start(start, stop)

	// Create timing objects on heap (remember to destroy!)
	CuBLASTimings* cublasTimings = new CuBLASTimings();
	CudaTimings* cudaTimings = new CudaTimings();

	for (int epoch = 1; epoch <= MAX_EPOCHS; ++epoch) {

		if (!cudaRun) {
			// Perform an iteration using the cuBLAS codepath
			cublas_iteration(
				cublasTimings,
				data_d,
				labels_d,
				data_shuffled_d,
				labels_shuffled_d,
				gradients,
				weights,
				loss_derivative,
				col_sums,
				ind_vector,
				batch_indices_d,
				data_raw_ptr,
				labels_raw_ptr,
				weights_raw_ptr,
				loss_derivative_raw_ptr,
				gradients_raw_ptr,
				learning_rate,
				epoch,
				num_batches,
				batchsize,
				R,
				C);
		} else {
			// Perform an iteration using the plain CUDA codepath
			cuda_iteration(
			    cudaTimings,
			    ind_vector,
			    gradients,
			    weights,
			    row_sums,
			    row_indices,
			    batch_indices_d,
			    data_raw_ptr,
			    labels_raw_ptr,
			    weights_raw_ptr,
			    gradients_raw_ptr,
			    batch_indices_ptr,
			    THREADS_PER_BLOCK,
			    learning_rate,
			    epoch,
			    num_batches,
			    batchsize,
			    R,
			    C);
		}


	}

	// Get the total GPU time
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float gpu_time = 0;
	cudaEventElapsedTime(&gpu_time, start, stop);

	if (!cudaRun) {
		printf("\nCUBLAS execution: \n");
	} else {
		printf("\nPlain CUDA execution: \n");
	}

	printf("Data transfer time = %f ms\n", transfer_time);
	printf("GPU time = %f ms\n", gpu_time);

	float avg_abs_error = calculate_mean_abs_error_cublas(
				data_raw_ptr,
				labels_raw_ptr,
				weights_raw_ptr,
				R,
				C);

	// Write timings to JSON file
	ExperimentOutput exp;
	std::string filename = get_filename_from_path(filepath);
	if (!cudaRun) {
		printf("Total gradient time = %f ms\n", cublasTimings->total_gradient_time);
		printf("Shuffle time = %f ms\n", cublasTimings->shuffle_time);
		printf("\n CUBLAS-specific timings: \n");
		printf("Derivative time = %f ms\n", cublasTimings->derivative_time);
		printf("Matrix scale time = %f ms\n", cublasTimings->matrix_scale_time);
		printf("Col sum/scale time = %f ms\n", cublasTimings->column_sum_time);
		printf("saxpy time = %f ms\n", cublasTimings->saxpy_time);

		exp = ExperimentOutput(cublasTimings->shuffle_time, cublasTimings->total_gradient_time, gpu_time, transfer_time, avg_abs_error);
		write_experiment_output(exp, filename + "-cublas.json");
	} else {
		printf("Total gradient time = %f ms\n", cudaTimings->total_gradient_time);
		printf("Shuffle time = %f ms\n", cudaTimings->shuffle_time);
		printf("\n Plain-CUDA-specific timings: \n");
		printf("gradient matrix time = %f ms\n", cudaTimings->grad_mat_time);
		printf("gradient vec. scale time = %f ms\n", cudaTimings->gradient_scale_time);
		printf("row sum time = %f ms\n", cudaTimings->row_sum_time);
		printf("saxpy time = %f ms\n", cudaTimings->saxpy_time);

		exp = ExperimentOutput(cudaTimings->shuffle_time, cudaTimings->total_gradient_time, gpu_time, transfer_time, avg_abs_error);

		write_experiment_output(exp, filename + "-cuda.json");
	}

	print_vector(weights, "weights");

	std::cout << "Mean absolute error : " << avg_abs_error << std::endl;

	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	delete cublasTimings;
	delete cudaTimings;

	return 0;
}
