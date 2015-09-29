
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

typedef std::vector<std::string> StringVector;

/* Split a line of strings into tokens and place them in a vector */
std::vector<std::string> split_into_tokens(std::string line)
{
    std::vector<std::string> result;

    std::stringstream linestream(line);
    std::string cell;

    while(std::getline(linestream,cell,','))
    {
        result.push_back(cell);
    }
    return result;
}

/* Read a csv file into a vector of string vectors.
 * Each item of the container vector is a vector of strings, with
 * each item in that vector being a token in a line in the csv file.
 */
std::vector<StringVector> read_csv(std::string filename)
{
	std::string line;
	std::vector<StringVector> string_vectors;
	// Open file as stream
	std::ifstream file_stream(filename);

	// Get lines and split into tokens. Push resulting vectors of tokens into container vector
	while (std::getline(file_stream, line))
	{
		string_vectors.push_back(split_into_tokens(line));
	}

	file_stream.close();

	return string_vectors;

}

// TODO: Conversion of StringVector to numerical
// TODO: Separate class label from features (i.e. take first or last element of vector)

// Read an example csv file and print it out
int main(int argc, char **argv) {

	std::string line;
	std::vector<StringVector> string_vectors;
	// TOFIX: Create a simple csv file named example.csv in the project dir to test
	string_vectors = read_csv("example.csv");

	for (auto str_vector : string_vectors)
	{
		for (StringVector::const_iterator i = str_vector.begin(); i != str_vector.end(); ++i) {
			std::cout << *i ;
			if (i != (str_vector.end() - 1)) { // Print out comma if not last element
				std::cout << ',';
			}
		}
		std::cout << std::endl;
	}

	return 0;
}
