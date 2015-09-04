#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

//////////////
// Declarations
//////////////

void print_vector(float *y, int N);
float linear_model(float x, float w0, float w1);
float batch_error(float *y_predic, float *y, int start, int end);
float gradient_error(float *y_predic, float *y, float *x, int start, int end, bool constant);
void evaluate_batch(float *y, float *x, int start, int end, float w0, float w1);

//////////////
// Main
//////////////

int main(){
   
  // System parametefunrs
  int i;
  int j;
  bool print = false;
  bool verbose = false;
    
  // First the main parameters
  int Nexamples = 1000;
  int Nbatch = 10;
  int batch_size = Nexamples / Nbatch;
  float w0 = 1;
  float w1 = 0;

  // Now the examples set
  float x[Nexamples];
  float y[Nexamples];
  float y_predic[Nexamples];

  // We get x
  for(i=0; i<Nexamples; i++){
    x[i] = (float) rand() / (RAND_MAX);
  }
  // We get y
  evaluate_batch(y, x, 0, Nexamples, w0, w1);

  // Now the algorithm
  int Niterations = 10000;
  float gamma = 0.1;
  float error0, error1;
  int start;
  int end; 
  float w0_est = 0.5;
  float w1_est = 0.5;
    
  for(j=0; j<Niterations; j++){
    for(i=0; i<Nbatch; i++){
      // Get the bounds of the batch
      start = i * batch_size;
      end = (i + 1) * batch_size;
      // Get errors and estimate
      evaluate_batch(y_predic, x, start, end, w0_est, w1_est);
      error0 = gradient_error(y_predic, y, x, start, end, true);
      error1 = gradient_error(y_predic, y , x, start, end, false);
      w0_est -= gamma * error0;
      w1_est -= gamma * error1;
      // Print messages if verbose
      if(verbose){
	printf("The start and end are %d %d \n", start, end);
	printf("The calculated errors are %f %f \n", error0, error1);
	printf("The estimated values for w are %f %f \n", w0_est, w1_est);
      }
    }
    if(verbose){
      printf("------------ \n");
    }
  }

  // Print the real and the estimated values
  printf("The real values for w are %f %f \n", w0, w1);
  printf("The estimated values for w are %f %f \n", w0_est, w1_est);

    
  // print this
  if(print){
    print_vector(y, Nexamples);
    print_vector(x, Nexamples);
    print_vector(y_predic, Nexamples);
  }

  return 0;
}

//////////////
// Functions
///////////////

void print_vector(float *y, int N){
  int i;
  for(i=0; i<N; i++){
    printf("for i = %d the value of y = %f \n", i, y[i]);
  }
  printf("---------- \n");
}


float linear_model(float x, float w0, float w1){
    return w0 + w1 * x;
}

float batch_error(float *y_predic, float *y, int start, int end){
  float error = 0;
  float aux; 
  int i = 0;

  for(i=start; i < end; i++){
    aux = (y_predic[i] - y[i]);
    error += aux * aux;
  }
  
  float size = (float) (end - start + 1);
  printf("error and size %f, %f \n", error, size);
  error = error / size;
  return error;
}

void evaluate_batch(float *y, float *x, int start,
		     int end, float w0, float w1){
  int i;
  
  for(i=start; i<end; i++){
    y[i] = linear_model(x[i], w0, w1);
  }
}


float gradient_error(float *y_predic, float *y, float *x, int start, int end, bool constant){
  float error = 0;
  int i = 0;

  if(constant){
    for(i=start; i < end; i++){
      error += (y_predic[i] - y[i]);
    }
  }
  else{
    for(i=start; i < end; i++){
      error += (y_predic[i] - y[i]) * x[i];
    }
  }
  
  float size = (float) (end - start + 1);
  error = error / size;
  return error;
}
