/*
 * sgd_thrust.cuh
 *
 *  Created on: Oct 26, 2015
 *      Author: tvas
 */

#ifndef SGD_THRUST_CUH_
#define SGD_THRUST_CUH_

#include <thrust/device_vector.h>
typedef thrust::device_vector<float> thrust_dev_float;
typedef thrust::device_vector<int> thrust_dev_int;

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
	float * gradients_d,
	const int R,
	const int C);

__host__ void calculate_row_sums(
	const int R,
	const int C,
	const thrust_dev_float& array,
	thrust_dev_float& row_sums,
	thrust_dev_int& row_indices);


#endif /* SGD_THRUST_CUH_ */
