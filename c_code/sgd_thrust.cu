#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

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
typedef thrust::host_vector<float> thrust_host_float;
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

//__device__ void print_device_vector(const float * vector, const int size, const char* name) {
//	printf("%s", name);
//	for (int i = 0; i < size; ++i) {
//		printf("%s ", vector[i]);
//	}
//	printf(" ]\n");
//}

template<typename T>
__host__ void print_vector(const T rowvector, const std::string name) {
	std::cout << name << " = [ ";
	for(auto element : rowvector)
	{
		std::cout << element << " ";
	}
	std::cout << "]" << std::endl;
}

template<typename T>
__host__ void print_matrix(const T matrix, const std::string name, const int R, const int C) {
	std::cout << name << std::endl;
	for(int i = 0; i < R; i++)
	{
		std::cout << "[ ";
		for(int j = 0; j < C; j++)
		  std::cout << matrix[i * C + j] << " ";
		std::cout << "]" << std::endl;
	}
}

/**
 * Reads a csv file of floats into two thrust host vectors passed by reference.
 * R should be the number of rows TODO: Do I want this? These we can also automatically figure out by ex. looking at the first line.
 * C should be the number of features available
 * The label should be the last column of the csv (i.e. C+1 with 1-indexing, C with 0-indexing)
 */
void read_csv(std::string filename, thrust_host_float & data_h, thrust_host_float & labels_h, int R, int C)
{
	// Open file as stream
	std::ifstream file_stream(filename);
	std::string line;

	// Get lines and split into tokens. Push resulting items into host vectors
	int row = 0, col = 0;
	while (std::getline(file_stream, line))
	{
		std::stringstream linestream(line);
		std::string cell;
		col = 0;
		while(std::getline(linestream,cell,','))
		{
			// Put elements to data vector, if at last column, push to labels instead
			float val = std::stof (cell);
			if (col != C) {
				data_h[row * C + col] = val;
			} else {
				labels_h[row] = val;
			}
			++col;
		}
		++row;
	}

	assert (labels_h.size() == R);
	assert (data_h.size() == R*C);

	file_stream.close();
}

__device__ float squared_loss_derivative(
	const float * data_array_d,
	const float * label_vector_d,
	const float * weights_d,
	const int C) {

	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	int example_index = thread_index * C;
//		printf("example_index: %d\n", example_index);
	// Calculate prediction
	float prediction = 0.0;
	// TODO: Guard against accesses that go outside the data_array_d limits
	for (int i = 0; i < C; ++i) {
		prediction += data_array_d[example_index + i] * weights_d[i];
	}
//		if (thread_index <= 1) {
//			printf("prediction in %d: %f\n", thread_index, prediction);
//		}
	// For squared loss the loss derivative is equal to the simple loss: (y_hat - y_true)
	float loss_derivative = prediction - label_vector_d[thread_index];
//		if (thread_index <= 1) {
//			printf("loss_deriv in %d: %f\n", thread_index, loss_derivative);
//		}
	//	float squared_loss = loss_derivative * loss_derivative;
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
	if (thread_index < R) {
		float loss_derivative = squared_loss_derivative(data_array_d, label_vector_d, weights_d, C);
		errors[thread_index] = loss_derivative * loss_derivative;
	}
}

__global__ void calculate_gradients(
	const float * data_array_d,
	const float * label_vector_d,
	const float * weights_d,
	float * gradients_d,
	const int R,
	const int C) {
	// TODO: R, C should be device constants
	// Each data point should get one thread
	// TODO: Verify that the accesses are correct, I think we modify the same elements
	// in the gradients matrix in different threads right now, which shouldn't happen.
	// We want each thread to take one "row" of the matrix and only modify that.
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	int example_index = thread_index * C;

	// TODO: Vector operations should be done with cuBLAS using dynamic parallelism
	if (thread_index < R) {

		float loss_derivative = squared_loss_derivative(data_array_d, label_vector_d, weights_d, C);
		// Scale the gradient by the loss_derivative
		for (int i = 0; i < C; ++i) {
			// For linear models the gradient equals the features
			gradients_d[example_index + i] = loss_derivative * data_array_d[example_index + i];
		}
	}
}

__host__ void calculate_row_sums(const int R, const int C, const thrust_dev_float& array, thrust_dev_float& row_sums, thrust_dev_int& row_indices) {
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

// TODO: Combine all iteration steps into one function
//__host__ void SGDStep(int R, int C, float * data_array_d, )

// Usage: Run with all arguments:
// args: [learning_rate] [iterations] [data_csv_file] [num_rows] [num_features]
// NB: We are assuming that the csv has the format [features],[label]
// i.e. the last column is the label, and all others are features.
// num_features should equal the number of features only, i.e. the number of columns in the csv minus 1
// e.g.: > ./sgd_thrust.o 0.001 10 data/5xy.csv 40 1
// or with real data: > ./sgd_thrust.o 0.000004 500 data/ENB2012_data_Y1.csv 768 8
int main(int argc, char **argv) {

	if	(argc != 6) {
		std::cout << "usage: ./sgd_thrust.o [learning_rate] "
				"[iterations] [data_csv_file] [num_rows] [num_features]" << std::endl;
	}

	float learning_rate = atof(argv[1]);
	int MAX_ITERATIONS = atoi(argv[2]);
	std::string filename = argv[3];
	int R = atoi(argv[4]);;
	int C = atoi(argv[5]);;

	std::cout << "lr: " << learning_rate << std::endl;

	// The number of threads we allocate per block
	const int THREADS_PER_BLOCK = 256;

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
	thrust_dev_float gradients(R * C);
	float * gradients_raw_ptr = thrust::raw_pointer_cast(gradients.data());

	//Initialize errors vector
	thrust_dev_float errors(R);
	float * errors_raw_ptr = thrust::raw_pointer_cast(errors.data());

	// Allocate storage for row sums and indices
	thrust_dev_float row_sums(R);
	thrust_dev_int row_indices(R);

	// Print out data and labels
//	print_matrix(data_d, "data", R, C);
//	print_vector<thrust_dev_float>(labels_d, "labels");

	for (int iteration = 1; iteration <= MAX_ITERATIONS; ++iteration) {
		// Reset gradients and errors
		thrust::fill(gradients.begin(), gradients.end(), 0.0);
		thrust::fill(errors.begin(), errors.end(), 0.0);

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
		thrust::fill(row_indices.begin(), row_indices.end(), 0.0);
		calculate_row_sums(R, C, gradients, row_sums, row_indices);

		// Scale gradient sum vector
		thrust::for_each(row_sums.begin(), row_sums.end(), _1 / (float)R);

		//Update the weight vector
		float a = -(learning_rate / std::sqrt(iteration));
//		std::cout << "a: " << a << std::endl;
		// Thrust SAXPY
		thrust::transform(row_sums.begin(), row_sums.end(),  // input range #1
		                      weights.begin(),           // input range #2
		                      weights.begin(),           // output range
		                      a * _1 + _2);        // placeholder expression

		// Calculate the squared error for each data point
//		cudaDeviceSynchronize();
		squared_errors<<<iDivUp(R, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
						data_raw_ptr,
						labels_raw_ptr,
						weights_raw_ptr,
						errors_raw_ptr,
						R,
						C);
		// Reduce/sum the errors
		float sq_err_sum = thrust::reduce(errors.begin(), errors.end());
		// Print weights and squared error sum
		print_vector<thrust_dev_float>(weights, "weights");
		std::cout << "Squared error sum: " << sq_err_sum << std::endl;
//		print_matrix(gradients, "weight_gradients", R, C);
	}


	return 0;
}
