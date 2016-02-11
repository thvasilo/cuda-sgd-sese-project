#include "testing.cuh"
#include "typedefs.cuh"
#include "sgd_io.cuh"
#include "sgd_thrust.cuh"

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

using namespace thrust::placeholders;

void test_abs_error() {
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
	float * loss_raw_ptr = thrust::raw_pointer_cast(loss_derivative.data());

	float avg_abs_loss = calculate_avg_loss_cublas(
		data_raw_ptr,
		labels_raw_ptr,
		weights_raw_ptr,
		loss_raw_ptr,
		R,
		C);

	std::cout << "Average absolute loss (8.0 expected): " << avg_abs_loss << std::endl;
}

void test_permutation() {
	const int R = 5;
	const int C = 4;

	// Initialize data vector on host
	thrust_host_float data_h(R * C);

	// Initialize labels vector on host
	thrust_host_float labels_h(R);

	std::string filename = "data/permutation_test.csv";
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

void test_col_sum_and_scale() {
	const int R = 5;
	const int C = 4;
	const float scaling_factor = 2.0;

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
