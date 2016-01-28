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
	const thrust_dev_float & matrix,
	const thrust_dev_float & scaling_vector,
	thrust_dev_float & matrix_normalized,
	const int C);

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


#endif /* SGD_THRUST_CUH_ */
