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
#include <ctime>

#include "json/json.h"

#include "typedefs.cuh"

/**
 * Container class for an experiment run, holding the timings for the various events.
 * TODO: Add parameter settings to (e.g. dataset name etc.?), creating a nested JSON object?
 */
class ExperimentOutput {

private:
	float shuffle_time, derivative_time, total_gradient_time, gpu_time, transfer_time;

public:
	ExperimentOutput(float _shuffle_time, float  _derivative_time, float  _total_gradient_time, float  _gpu_time, float  _transfer_time):
		shuffle_time(_shuffle_time),
		derivative_time(_derivative_time),
		total_gradient_time(_total_gradient_time),
		gpu_time(_gpu_time),
		transfer_time(_transfer_time) {}

	Json::Value toJsonObject() const {
		Json::Value jsonObj;

		jsonObj["shuffle_time"] = this->shuffle_time;
		jsonObj["derivative_time"] = this->derivative_time;
		jsonObj["total_gradient_time"] = this->total_gradient_time;
		jsonObj["gpu_time"] = this->gpu_time;
		jsonObj["transfer_time"] = this->transfer_time;

		return jsonObj;
	}
};

void print_vector(const thrust_host_float rowvector, const std::string name);

void print_int_vector(const thrust_dev_int rowvector, const std::string name);

void print_matrix(const thrust_host_float matrix, const std::string name, const int R, const int C);

/**
 * Writes a Json object to the provided filepath, using pretty formatting.
 */
void write_json(Json::Value json_object, std::string filepath);

/**
 * Writes experiment timing results to JSON file.
 */
void write_experiment_output(ExperimentOutput exp, std::string filepath);

/**
 * Returns the name of a file from a filepath, excluding the type suffix.
 * Example: get_filename_from_path("/path/to/file.csv") should return "file"
 */
std::string get_filename_from_path(std::string filepath);

/**
 * Reads a csv file of floats into two thrust host vectors passed by reference.
 * R should be the number of rows TODO: Do I want this? These we can also automatically figure out by
 * ex. looking at the first line.
 * C should be the number of features available
 * The label should be the last column of the csv (i.e. C+1 with 1-indexing, C with 0-indexing)
 */
void read_csv(std::string filename, thrust_host_float & data_h, thrust_host_float & labels_h, int R, int C);


#endif /* SGD_IO_CUH_ */
