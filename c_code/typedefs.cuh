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

// Macro to combine all the calls for stopping, measuring and printing an event, and maintaining a sum
// TODO: Add string for name, maybe return elapsed time?
#define measure_event(start_event, stop_event, sum, name) { \
	cudaEventRecord(stop_event); \
	cudaEventSynchronize(stop_event); \
	float miliseconds_reference = 0; \
	cudaEventElapsedTime(&miliseconds_reference, start_event, stop_event); \
	sum += miliseconds_reference; \
	cudaEventDestroy(start_event); \
	cudaEventDestroy(stop_event); \
	} // 	printf("%s time = %f ms\n", name, miliseconds_reference); \

#define create_events_and_start(start_event, stop_event) { \
	cudaEventCreate(&start_event); \
	cudaEventCreate(&stop_event); \
	cudaEventRecord(start_event); \
}

// #define LVL1 // Used to control whether to perform dynamic parallelism calls in the plain CUDA codepath.

#endif /* TYPEDEFS_CUH_ */
