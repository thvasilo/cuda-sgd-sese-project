#include "sgd_thrust.cuh"


// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
  T C; // number of columns

  __host__ __device__
  linear_index_to_row_index(T C) : C(C) {}

  __host__ __device__
  T operator()(T i)
  {
    return i / C;
  }
};


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

	// Calculate prediction
	float prediction = 0.0;
	// TODO: Guard against accesses that go outside the data_array_d limits
	// TODO: Vector operations should be done with cuBLAS using dynamic parallelism
	for (int i = 0; i < C; ++i) {
		prediction += data_array_d[i] * weights_d[i];
	}
	// For squared loss the loss derivative is equal to the simple loss: (y_hat - y_true)
	float loss_derivative = prediction - *label_vector_d;
	return loss_derivative;
}

__global__ void squared_errors(
	const float * data_array_d,
	const float * label_vector_d,
	const float * weights_d,
	float * errors,
	const int R,
	const int C) {

	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	int example_index = thread_index * C;
	if (thread_index < R) {
		float loss_derivative = squared_loss_derivative(
				&data_array_d[example_index],
				&label_vector_d[thread_index],
				weights_d,
				C);
		errors[thread_index] = loss_derivative * loss_derivative;
	}
}

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

	int thread_index = blockIdx.x * blockDim.x + threadIdx.x; // Should be row index
//	int example_index = thread_index * C;

	// TODO: Vector operations should be done with cuBLAS using dynamic parallelism
	if (thread_index < R) {
		int example_index = batch_indices_d[thread_index] * C;
//		printf("thread idx: %d, example idx: %d\n", thread_index, example_index);
		// Here we call squared_loss_derivative with the correct offsets, example_index for the data array
		// and thread_index for the labels array
		float loss_derivative = squared_loss_derivative(
				&data_array_d[example_index],
				&label_vector_d[thread_index],
				weights_d,
				C);
		// Scale the gradient by the loss_derivative
		for (int i = 0; i < C; ++i) {
			// For linear models the gradient equals the features
			gradients_d[example_index + i] = loss_derivative * data_array_d[example_index + i];
		}
	}
}

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
