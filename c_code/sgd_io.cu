#include "sgd_io.cuh"

void print_vector(const thrust_host_float rowvector, const std::string name) {
	std::cout << name << " = [ ";
	for(auto element : rowvector)
	{
		std::cout << element << " ";
	}
	std::cout << "]" << std::endl;
}

void print_int_vector(const thrust_dev_int rowvector, const std::string name) {
	std::cout << name << " = [ ";
	for(auto element : rowvector)
	{
		std::cout << element << " ";
	}
	std::cout << "]" << std::endl;
}

void print_matrix(const thrust_host_float matrix, const std::string name, const int R, const int C) {
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

/**
 * Writes a JSON object to a file in the provided filepath.
 */
void write_json(Json::Value json_object, std::string filepath) {
	std::ofstream json_file(filepath);

	Json::StyledWriter styledWriter;

	json_file << styledWriter.write(json_object);

	json_file.close();

}

void write_experiment_output(ExperimentOutput exp, std::string filepath) {
//	// time_t to string lambda
//	auto timestr = [](time_t t) -> std::string {
//	   std::stringstream strm;
//	   strm << t;
//	   return strm.str();
//	};
//
//	std::time_t t = std::time(nullptr);
//	std::string timestamp = timestr(t);
//	std::string file_with_time = filepath + "-" + timestamp;
	write_json(exp.toJsonObject(), filepath);
}

std::string get_filename_from_path(std::string const filepath) {
	auto pos = filepath.find_last_of('/');
	auto filename = filepath.substr(pos+1);
	pos = filename.find_last_of('.');
	filename = filename.substr(0, pos);
	return filename;
}

//__device__ void print_device_vector(const float * vector, const int size, const char* name) {
//	printf("%s", name);
//	for (int i = 0; i < size; ++i) {
//		printf("%s ", vector[i]);
//	}
//	printf(" ]\n");
//}
