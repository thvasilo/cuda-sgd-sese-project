#include "testing.cuh"
#include "typedefs.cuh"
#include "sgd_io.cuh"
#include "sgd_thrust.cuh"

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

using namespace thrust::placeholders;

// TODO: Automate testing

void test_abs_error() {
	const int R = 5;
	const int C = 4;

	// Initialize data vector on host
	thrust_host_float data_h(R * C);

	// Initialize labels vector on host
	thrust_host_float labels_h(R);

	std::string filename = "test-data/gemv_test";
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

	float avg_abs_loss = calculate_mean_abs_error_cublas(
		data_raw_ptr,
		labels_raw_ptr,
		weights_raw_ptr,
		R,
		C);

	std::cout << "Mean absolute error (8.0 expected): " << avg_abs_loss << std::endl;
}

void test_permutation() {
	const int R = 5;
	const int C = 4;

	// Initialize data vector on host
	thrust_host_float data_h(R * C);

	// Initialize labels vector on host
	thrust_host_float labels_h(R);

	std::string filename = "test-data/permutation_test.csv";
	// Read data from csv file into host vectors
	read_csv(filename, data_h, labels_h, R, C);

	print_matrix(data_h, "data_h", R, C);
	print_vector(labels_h, "labels");

	// Copy data from host vectors to device
	thrust_dev_float data_d = data_h;
	thrust_dev_float labels_d = labels_h;

	// Initialize the order vector
	thrust::device_vector<int> order(R);
	order[0] = 4;
	order[1] = 2;
	order[2] = 0;
	order[3] = 1;
	order[4] = 3;

	// Storage for the permuted data
	thrust_dev_float permuted_data(R*C);
	thrust_dev_float permuted_labels(R);

	permute_data_and_labels(
			data_d,
			labels_d,
			order,
			permuted_data,
			permuted_labels,
			R,
			C);

	// Copy back from device
	data_h = permuted_data;
	labels_h = permuted_labels;

	print_matrix(data_h, "permuted data_h", R, C);
	print_vector(labels_h, "permuted labels_h");
}

void test_col_sums() {
	const int R = 5;
	const int C = 4;

	// Initialize data vector on host
	thrust_host_float data_h(R * C);

	// Initialize labels vector on host
	thrust_host_float labels(R);

	std::string filename = "test-data/col_sum_test.csv";
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

void test_col_sum_and_scale() {
	const int R = 5;
	const int C = 4;
	const float scaling_factor = 2.0;

	// Initialize data vector on host
	thrust_host_float data_h(R * C);

	// Initialize labels vector on host
	thrust_host_float labels(R);

	std::string filename = "test-data/col_sum_test.csv";
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

	// Scale the column sums
	thrust::for_each(col_sums.begin(), col_sums.end(), _1 / scaling_factor);
	col_sums_h = col_sums;
	print_vector(col_sums_h, "scaled_column_sums");

	// Perform the fused column sum and scale
	calculate_scaled_col_sums(
		data_dev_ptr,
		col_sums, // col_sums will now contain the sum of the columns in the gradient matrix, scaled by 1/batchsize
		R,
		C,
		scaling_factor);

	col_sums_h = col_sums;
	print_vector(col_sums_h, "fused_scaled_column_sums");
}

void test_matrix_scale() {
	const int R = 5;
	const int C = 4;

	// Initialize data vector on host
	thrust_host_float data_h(R * C);

	// Initialize labels vector on host
	thrust_host_float scaling_vector(R);

	std::string filename = "test-data/matrix_scale_test";
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

	std::string filename = "test-data/gemv_test";
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

// The following taken from https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/

#include <stdio.h>
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %sn",
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void profileCopies(float        *h_a,
                   float        *h_b,
                   float        *d,
                   unsigned int  n,
                   char         *desc)
{
  printf("\n%s transfers\n", desc);

  unsigned int bytes = n * sizeof(float);

  // events for timing
  cudaEvent_t startEvent, stopEvent;

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  float time;
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  for (int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** %s transfers failed ***", desc);
      break;
    }
  }

  // clean up events
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}

void check_transfer_speed()
{
  unsigned int nElements = 4*1024*1024*16;
  const unsigned int bytes = nElements * sizeof(float);

  // host arrays
  float *h_aPageable, *h_bPageable;
  float *h_aPinned, *h_bPinned;

  // device array
  float *d_a;

  // allocate and initialize
  h_aPageable = (float*)malloc(bytes);                    // host pageable
  h_bPageable = (float*)malloc(bytes);                    // host pageable
  checkCuda( cudaMallocHost((void**)&h_aPinned, bytes) ); // host pinned
  checkCuda( cudaMallocHost((void**)&h_bPinned, bytes) ); // host pinned
  checkCuda( cudaMalloc((void**)&d_a, bytes) );           // device

  for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;
  memcpy(h_aPinned, h_aPageable, bytes);
  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);

  // output device info and transfer size
  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, 0) );

  printf("\nDevice: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

  // perform copies and report bandwidth
  profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
  profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

  printf("\n");

  // cleanup
  cudaFree(d_a);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable);
  free(h_bPageable);
}
