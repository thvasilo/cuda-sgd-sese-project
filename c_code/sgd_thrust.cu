#include "sgd_thrust.cuh"
#include "sgd_io.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/sequence.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h>

/**
 * Returns the squared loss derivative
 * The way this works is: We provide pointers to items inside the data and label vectors,
 * and we proceed to take the C next elements from the data array and the weights,
 * to calculate their inner product. We then dereference the pointer to the label vector
 * to get the value at the position.
 */
__device__ float squared_loss_derivative(
	const float * data_array_d,
	const float * label_vector_d,
	const float * weights_d,
	const int C) {
	cublasHandle_t cublasHandle;
	cublasStatus_t status = cublasCreate(&cublasHandle);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("Something went wrong with cuBLAS initialization in squared_loss_derivative\n");
	}

	// Calculate prediction
	float prediction = 0.0;
	float * pred_p = &prediction;
	// TODO: Guard against accesses that go outside the data_array_d limits
	cublasSdot(cublasHandle, C, data_array_d, 1, weights_d, 1, pred_p);
//	for (int i = 0; i < C; ++i) {
//		prediction += data_array_d[i] * weights_d[i];
//	}
	// For squared loss the loss derivative is equal to the simple loss: (y_hat - y_true)
	float loss_derivative = prediction - *label_vector_d;

	cublasDestroy(cublasHandle);
	return loss_derivative;
}

/**
 * Calculates the sum of squared errors, given the data, labels and weights
 */
__global__ void squared_errors(
	const float * data_array_d,
	const float * label_vector_d,
	const float * weights_d,
	float * errors,
	const int R,
	const int C) {

	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	int linear_index = thread_index * C;
	if (thread_index < R) {
		float loss_derivative = squared_loss_derivative(
				&data_array_d[linear_index],
				&label_vector_d[thread_index],
				weights_d,
				C);
		errors[thread_index] = loss_derivative * loss_derivative * 0.5;
	}
}

/**
 * Calculates the weight gradients for each data point in the batch
 */
__global__ void calculate_gradients(
	const float * data_array_d,
	const float * label_vector_d,
	const float * weights_d,
	const int * batch_indices_d,
	float * gradients_d,
	const int R,
	const int C) {
	// TODO: R, C should be device constants
	// Each data point should get one thread
	// We want each thread to take one "row" of the matrix and only modify that.
	cublasHandle_t cnpHandle;
	cublasStatus_t status = cublasCreate(&cnpHandle);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("Something went wrong with cuBLAS initialization in calculate_gradients\n");
	}

	int thread_index = blockIdx.x * blockDim.x + threadIdx.x; // Should be row index

	// TODO: Vector operations should be done with cuBLAS using dynamic parallelism
	if (thread_index < R) {
		// Example index is the current example that we are examining in the batch, and is in [0, R_all-1], where R_all the total number
		// data points
		int example_index = batch_indices_d[thread_index];
		// The linear index gives us the first element of the data array for the example in the interleaved array.
		int linear_index = example_index * C;

		// Here we call squared_loss_derivative with the correct offsets, example_index for the data array
		// and thread_index for the labels array
		float loss_derivative = squared_loss_derivative(
				&data_array_d[linear_index],
				&label_vector_d[example_index],
				weights_d,
				C);
		// Scale the gradient by the loss_derivative
		for (int i = 0; i < C; ++i) {
			// For linear models the gradient equals the features
			gradients_d[i*R + thread_index] = loss_derivative * data_array_d[linear_index + i];
		}
	}

	cublasDestroy(cnpHandle);
}

/**
 * Calculates the sum of the rows in an interleaved array using Thrust
 */
__host__ void calculate_row_sums(
	const int R,
	const int C,
	const thrust_dev_float & array,
	thrust_dev_float & row_sums,
	thrust_dev_int & row_indices) {
	// compute row sums by summing values with equal row indices
	thrust::reduce_by_key(
		thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)),
		thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)) + (R*C),
		array.begin(),
		row_indices.begin(),
		row_sums.begin(),
		thrust::equal_to<int>(),
		thrust::plus<float>());
}

// Summation functor used for the column sum operation
struct sum_functor
{
  int R;
  int C;
  float *arr;

  sum_functor(int _R, int _C, float *_arr) : R(_R), C(_C), arr(_arr) {};

  __host__ __device__
  float operator()(int myC){
	  float sum = 0.0;
	  for (int i = 0; i < R; ++i) {
		  sum += arr[i*C+myC];
	  }
	return sum;
	}
};

/**
 * Calculates the sum of each column in array (of size R*C) and stores it in col_sums.
 */
__host__ void calculate_column_sums(
		float * array,
		thrust_dev_float& col_sums,
		const int R,
		const int C) {

	// Need to initially set the elements of the sums vector to a sequence, since they represent the indices.
	thrust::sequence(col_sums.begin(), col_sums.end());
	// Perform the sum column transformation
	thrust::transform(
		col_sums.begin(),
		col_sums.end(),
		col_sums.begin(),
		sum_functor(R, C, array));
}

/**
 * For an RxC matrix and a vector of size R, scales each element in each row of the matrix
 * by the corresponding element in the scaling_vector.
 * i.e. each element in row 1 of the matrix is multiplied by the element 1 of the scaling_vector,
 * each element in row 2 is multiplied by element 2 ine the vector etc.
 * The result is stored in matrix_normalized.
 * Taken from: http://stackoverflow.com/q/9290935/209882
 */
__host__ void scale_matrix_rows_by_vector(
	const thrust::device_ptr<float> & matrix,
	const thrust_dev_float & scaling_vector,
	thrust_dev_float & matrix_normalized,
	const int R,
	const int C) {

	thrust::transform(
		matrix, matrix + (R*C),
	    thrust::make_permutation_iterator(
	    scaling_vector.begin(),
	    thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), linear_index_to_row_index<int>(C))),
	    matrix_normalized.begin(),
	    thrust::multiplies<float>());
}

// The loss derivative vector is equal to the vector of predictions minus the label vector for the batch.
// We get the prediction vector from the matrix-vector multiplication of the data batch matrix with the
// weights vector, from which we then subtract the labels vector, which we have previously copied from
// label_vector into the loss derivative vector, which is the destination for the complete operation.
// In short: loss_der = B * w - label
// where B is the batch data matrix and the rest are vectors
__host__ void calculate_loss_derivative_cublas(
	const float * data_array_d,
	const float * label_vector_d,
	const float * weights_d,
	float * loss_derivative_d,
	const int R,
	const int C,
	const int batchsize) {
	// We are assuming that data_array_d points to the first element in the batch in the data array,
	// and label_vector_d points to the corresponding element in the label vector

	// Set up cuBLAS environment
	cublasHandle_t cnpHandle;
	cublasStatus_t status = cublasCreate(&cnpHandle);

	// We copy the labels into the loss derivative vector, since the gemv operation below is destructive for
	// the destination vector
	cublasScopy(
			cnpHandle,
			batchsize,
			label_vector_d, 1,
			loss_derivative_d, 1);

	const float alpha = 1.0;
	const float a_minus = -1.0;

	// To use gemv with row-major matrices we enter reverse values for rows and columns, take the transpose
	// of the matrix and assign the number of columns as the leading dimension
	// See http://peterwittek.com/cublas-matrix-c-style.html and http://stackoverflow.com/q/21164373/209882
	status = cublasSgemv(
		cnpHandle,
		CUBLAS_OP_T,
		C, // Set the num of columns as the number of rows in the matrix
		batchsize, // Set the num of rows (=batchsize) as the number of columns in the matrix
		&alpha,
		data_array_d,
		C, // We set the leading dimension for data_array to be the number of cols, since we are using C-style row-major arrays.
		weights_d,
		1,
		&a_minus,
		loss_derivative_d, // The result of the calculation is stored in the loss derivative vector
		1);
	// Check if the gemv operation was successful
	if (status != CUBLAS_STATUS_SUCCESS) {
		    std::cerr << "Failed to execute the gemv!\n";
	}
	// Clean up cuBLAS environment
	cublasDestroy(cnpHandle);
}

// Permute matrix rows on device
__host__ void permute_data_and_labels(
		const thrust_dev_float& data,
		const thrust_dev_float& labels,
		const thrust::device_vector<unsigned>& order,
		thrust_dev_float& permuted_data,
		thrust_dev_float& permuted_labels,
		const unsigned R,
		const unsigned C) {

	// TODO: copy_n docs have an error(?): "Generally, for every integer i from 0 to n," should be "to n-1"?
	// permute the matrix
	thrust::copy_n(
		thrust::make_permutation_iterator(
		  data.begin(),
		  thrust::make_transform_iterator(
			thrust::counting_iterator<unsigned>(0),
			copy_idx_func(C, thrust::raw_pointer_cast(order.data())))),
		R*C, permuted_data.begin());

	// permute the vector
	thrust::copy_n(
		thrust::make_permutation_iterator(
		  labels.begin(),
		  thrust::make_transform_iterator(
			thrust::counting_iterator<unsigned>(0),
			copy_idx_func(1,thrust::raw_pointer_cast(order.data())))),
		R, permuted_labels.begin());

}
