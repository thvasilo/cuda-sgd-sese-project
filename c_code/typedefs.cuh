/*
 * typedefs.cuh
 *
 *  Created on: Oct 27, 2015
 *      Author: tvas
 */

#ifndef TYPEDEFS_CUH_
#define TYPEDEFS_CUH_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

typedef thrust::host_vector<float> thrust_host_float;
typedef thrust::device_vector<float> thrust_dev_float;
typedef thrust::device_vector<int> thrust_dev_int;


#endif /* TYPEDEFS_CUH_ */
