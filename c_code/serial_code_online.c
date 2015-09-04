#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

float linear_model(float x, float w0, float w1){
    return w0 + w1 * x;
}

int main(){

    // System parameters
    int i = 0;
    int j = 0;
    bool print = false;
    bool verbose = false;
    
    // First the main parameters
    int Nexamples = 100;
    float w0 = 3;
    float w1 = 10;

    // Now the examples set
    float x[Nexamples];
    float y[Nexamples];

    // Now we define x and y 
    for(i=0; i<Nexamples; i++){
      x[i] = (float) rand() / (RAND_MAX);
    }

    for(i=0; i<Nexamples; i++){
	y[i] = linear_model(x[i], w0, w1);
    }
       
    // print this
    if(print){
	for(i=0; i<Nexamples; i++){
	    printf("for i = %d the value of y = %f \n", i, y[i]);
	}
    }
   
    // print this
    if(print){
	for(i=0; i<Nexamples; i++){
	    printf("for i = %d the value of x = %f \n", i, x[i]);
	}
    }

    // Here goes the algorithm
    int Niter = 100;
    float gamma = 0.1;

    float w0_est = 0.5;
    float w1_est = 0.5;
    float y_est;
    float error;

    for(j=0; j<Niter; j++){
	for(i=0; i<Nexamples; i++){
	    y_est = linear_model(x[i], w0_est, w1_est);
	    error = gamma * (y_est - y[i]);
	    w0_est -= error;
	    w1_est -= error * x[i];
	    if(verbose){
	      printf("The estimated values for w are %f %f \n", w0_est, w1_est);
	      printf("the error is %f \n", error);	      
	    }
	}
	if(verbose){
	  printf("------------- \n");
	}
    }

    // Print the real and the estimated values
    printf("The real values for w are %f %f \n", w0, w1);
    printf("The estimated values for w are %f %f \n", w0_est, w1_est);

    return 0;
}
