#include "sampling.cuh"

#include <random>

double GetUniform()
{
	static std::default_random_engine re;
	static std::uniform_real_distribution<double> Dist(0,1);
	return Dist(re);
}

// Taken from John D. Cook, http://stackoverflow.com/a/311716/15485
void SampleWithoutReplacement
(
    int populationSize,    // size of set sampling from
    int sampleSize,        // size of each sample
    thrust_dev_int & samples  // output, zero-offset indices to selected items
)
{
    // Use Knuth's variable names
    int& n = sampleSize;
    int& N = populationSize;

    int t = 0; // total input records dealt with
    int m = 0; // number of items selected so far
    double u;

    while (m < n)
    {
        u = GetUniform(); // call a uniform(0,1) random number generator

        if ( (N - t)*u >= n - m )
        {
            t++;
        }
        else
        {
            samples[m] = t;
            t++; m++;
        }
    }
}

//#include <iostream>
//int main(int,char**)
//{
//  const size_t sz = 10;
//  thrust_dev_int samples(sz);
//  SampleWithoutReplacement(10*sz,sz,samples);
//  for (size_t i = 0; i < sz; i++ ) {
//    std::cout << samples[i] << "\t";
//  }
//  std::cout << std::endl;
//
//  return 0;
//}
