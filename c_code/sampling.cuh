/*
 * sampling.cuh
 *
 *  Created on: Oct 27, 2015
 *      Author: tvas
 */

#ifndef SAMPLING_CUH_
#define SAMPLING_CUH_

#include "typedefs.cuh"

double GetUniform();

void SampleWithoutReplacement
(
    int populationSize,    // size of set sampling from
    int sampleSize,        // size of each sample
    thrust_dev_int & samples  // output, zero-offset indices to selected items
);


#endif /* SAMPLING_CUH_ */
