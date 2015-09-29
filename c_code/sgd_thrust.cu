#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
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

typedef thrust::device_vector<float> thrust_dev_float;
typedef thrust::device_vector<int> thrust_dev_int;

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

__global__ void calculate_gradients(const float * data_array_d, const int * label_vector_d, const float * weights_d, float * gradients_d, const int R, const int C) {
	// TODO: R, C should be device constants
	// Each data point should get one thread
	// TODO: Verify that the accesses are correct, I think we modify the same elements
	// in the gradients matrix in different threads right now, which shouldn't happen.
	// We want each thread to take one "row" of the matrix and only modify that.
	int example_index = blockIdx.x * blockDim.x + threadIdx.x;

	// TODO: Vector operations should be done with cuBLAS using dynamic parallelism
	if (example_index < R) {
		printf("example_index: %d\n", example_index);
		// Calculate prediction
		float prediction = 0.0;
		// TODO: Guard against accesses that go outside the data_array_d limits
		for (int i = 0; i < C; ++i) {
			prediction += data_array_d[example_index + i] * weights_d[i];
		}
		float loss_derivative = label_vector_d[example_index] - prediction;
		//	float squared_loss = loss_derivative * loss_derivative;

		// Scale the prediction gradient by the loss_derivative
		for (int i = 0; i < C; ++i) {
			// For linear models the gradient equals the features
			gradients_d[example_index + i] = loss_derivative * data_array_d[example_index + i];
		}
	}
}

__host__ void calculate_row_sums(int R, int C, thrust_dev_float& array, thrust_dev_float& row_sums, thrust_dev_int& row_indices) {
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

__host__ void print_vector(thrust_dev_float rowvector, std::string name) {
	std::cout << name << " = [ ";
	for(auto element : rowvector)
	{
		std::cout << element << " ";
	}
	std::cout << "]" << std::endl;
}

// TODO: Combine all iteration steps into one function
//__host__ void SGDStep(int R, int C, float * data_array_d, )


int main(int argc, char **argv) {

	float learning_rate;
	if (argc == 1) {
		learning_rate = 0.001;
	} else {
		learning_rate = atof(argv[1]);
	}
	std::cout << "lr: " << learning_rate << std::endl;

	const int MAX_ITERATIONS = 10;
	const int THREADS_PER_BLOCK = 256;
	int R = 5;     // number of rows
	int C = 8;     // number of columns
	thrust::default_random_engine rng;
	thrust::uniform_real_distribution<float> dist(10, 99);
	thrust::uniform_real_distribution<float> weight_dist(0.0, 1.0);
	thrust::uniform_int_distribution<int> label_dist(0, 1);

	// Initialize data
	thrust_dev_float array(R * C);
	// note: d_vec.data() returns a device_ptr
	float * data_raw_ptr = thrust::raw_pointer_cast(array.data());
	for (size_t i = 0; i < array.size(); i++) {
		array[i] = dist(rng);
	}

	// Initialize labels
	thrust_dev_int labels(R);
	int * labels_raw_ptr = thrust::raw_pointer_cast(labels.data());
	for (size_t i = 0; i < labels.size(); i++) {
			labels[i] = label_dist(rng);
	}

	// Initialize weights
	thrust_dev_float weights(C);
	// note: d_vec.data() returns a device_ptr
	float * weights_raw_ptr = thrust::raw_pointer_cast(weights.data());
	for (size_t i = 0; i < weights.size(); i++) {
				weights[i] = weight_dist(rng);
	}

	// Initialize gradients
	thrust_dev_float gradients(R * C);
	// note: d_vec.data() returns a device_ptr
	float * gradients_raw_ptr = thrust::raw_pointer_cast(gradients.data());
	thrust::fill(gradients.begin(), gradients.end(), 0.0);

	// allocate storage for row sums and indices
	thrust_dev_float row_sums(R);
	thrust_dev_int row_indices(R);

	for (int iteration = 1; iteration <= MAX_ITERATIONS; ++iteration) {
		thrust::fill(gradients.begin(), gradients.end(), 0.0);
		//Calculate the gradient vector for each datapoint
		calculate_gradients<<<iDivUp(R, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
				data_raw_ptr,
				labels_raw_ptr,
				weights_raw_ptr,
				gradients_raw_ptr,
				R,
				C);

		// Sum/reduce the gradient vectors
		thrust::fill(row_sums.begin(), row_sums.end(), 0.0);
		calculate_row_sums(R, C, gradients, row_sums, row_indices);
		//TODO: Do I need to reset the sums to 0?
		// Can we re-use the iterators?
		// Scale gradient vector
		thrust::for_each(gradients.begin(), gradients.end(), _1 / (float)R);

		//Update the weight vector
		float a = -(learning_rate / std::sqrt(iteration));
		std::cout << "a: " << a << std::endl;
		// Thrust SAXPY
		thrust::transform(gradients.begin(), gradients.end(),  // input range #1
		                      weights.begin(),           // input range #2
		                      weights.begin(),           // output range
		                      a * _1 + _2);        // placeholder expression

		print_vector(weights, "weights");
		print_vector(gradients, "weight_gradient");
	}


	return 0;
}
