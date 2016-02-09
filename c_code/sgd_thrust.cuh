/*
 * sgd_thrust.cuh
 *
 *  Created on: Oct 26, 2015
 *      Author: tvas
 */

#ifndef SGD_THRUST_CUH_
#define SGD_THRUST_CUH_

#include "typedefs.cuh"
#include <cublas_v2.h>

__device__ float squared_loss_derivative(
	const float * data_array_d,
	const float * label_vector_d,
	const float * weights_d,
	const int C);

__global__ void squared_errors(
	const float * data_array_d,
	const float * label_vector_d,
	const float * weights_d,
	float * errors,
	const int R,
	const int C);

__global__ void calculate_gradients(
	const float * data_array_d,
	const float * label_vector_d,
	const float * weights_d,
	const int * batch_indices_d,
	float * gradients_d,
	const int R,
	const int C);

__host__ void calculate_row_sums(
	const int R,
	const int C,
	const thrust_dev_float& array,
	thrust_dev_float& row_sums,
	thrust_dev_int& row_indices);

__host__ void calculate_loss_derivative_cublas(
	const float * data_array_d,
	const float * label_vector_d,
	const float * weights_d,
	float * loss_derivative_d,
	const int R,
	const int C,
	const int batchsize);

__host__ void scale_matrix_rows_by_vector(
	const thrust::device_ptr<float> & matrix,
	const thrust_dev_float & scaling_vector,
	thrust_dev_float & matrix_normalized,
	const int R,
	const int c);

__host__ void calculate_column_sums(
		float * array, // TODO: Can I make this into a const?
		thrust_dev_float& col_sums,
		const int R,
		const int C);

__host__ void calculate_scaled_col_sums(
		const float * array,
		thrust_dev_float& col_sums,
		const int R,
		const int C,
		const float scaling_factor);

__host__ void permute_data_and_labels(
		const thrust_dev_float& data,
		const thrust_dev_float& labels,
		const thrust::device_vector<unsigned>& order,
		thrust_dev_float& permuted_data,
		thrust_dev_float& permuted_labels,
		const unsigned R,
		const unsigned C);


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

template <typename T>
struct row_index_to_linear_index : public thrust::unary_function<T,T>
{
  T C; // number of columns

  __host__ __device__
  row_index_to_linear_index(T C) : C(C) {}

  __host__ __device__
  T operator()(T i)
  {
    return i * C;
  }
};

// Used to permute matrix/vectors
// TODO: Investigate function
struct copy_idx_func : public thrust::unary_function<unsigned, unsigned>
{
  size_t c;
  const unsigned *p;
  copy_idx_func(const size_t _c, const unsigned *_p) : c(_c),p(_p) {};
  __host__ __device__
  unsigned operator()(unsigned idx){
    unsigned myrow = idx/c;
    unsigned newrow = p[myrow];
    unsigned mycol = idx%c;
    return newrow*c+mycol;
  }
};

#endif /* SGD_THRUST_CUH_ */
