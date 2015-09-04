#include <stdio.h>

float linear_model(float x, float w0, float w1){
    return w0 + w1 * x;
}


int main(){

    // System parameters
    int i = 0;
    int j = 0;
    
    // First the main parameters
    int Nexamples = 10;
    float w0 = 0;
    float w1 = 2;

    // Now the examples set
    float x[Nexamples];
    float y[Nexamples];

    // Now we define x and y 
    for(i=0; i<Nexamples; i++){
	x[i] = i;
    }

    for(i=0; i<Nexamples; i++){
	y[i] = linear_model(x[i], w0, w1);
    }
       

    // print this
    if(1 == 0){
	for(i=0; i<Nexamples; i++){
	    printf("for i = %d the value of y = %f \n", i, y[i]);
	}
    }

    // Here goes the algorithm
    int Niter = 10;
    float gamma = 0.1;

    float w0_est = 0.0;
    float w1_est = 2.0;
    float y_est;
    float error;


    for(j=0; j<Niter; j++){
	for(i=0; i<Nexamples; i++){
	    y_est = linear_model(x[i], w0_est, w1_est);
	    error = gamma * (y_est - y[i]);
	    printf("the error is %f \n", error);
	    w0_est = w0_est - error;
	    w1_est = w1_est - error * x[i];
	}
    }

    // Print the real and the estimated values
    printf("The real values for w are %f %f \n", w0, w1);
    printf("The estimated values for w are %f %f \n", w0_est, w1_est);

    return 0;
}
