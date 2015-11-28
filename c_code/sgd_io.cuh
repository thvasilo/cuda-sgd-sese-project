/*
 * sgd_io.cuh
 *
 *  Created on: Oct 26, 2015
 *      Author: tvas
 */

#ifndef SGD_IO_CUH_
#define SGD_IO_CUH_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

#include "typedefs.cuh"

void print_vector(const thrust_host_float rowvector, const std::string name);

void print_int_vector(const thrust_dev_int rowvector, const std::string name);

void print_matrix(const thrust_host_float matrix, const std::string name, const int R, const int C);

/**
 * Reads a csv file of floats into two thrust host vectors passed by reference.
 * R should be the number of rows TODO: Do I want this? These we can also automatically figure out by
 * ex. looking at the first line.
 * C should be the number of features available
 * The label should be the last column of the csv (i.e. C+1 with 1-indexing, C with 0-indexing)
 */
void read_csv(std::string filename, thrust_host_float & data_h, thrust_host_float & labels_h, int R, int C);


#endif /* SGD_IO_CUH_ */
