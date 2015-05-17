#include "create_filter.h"

float* gaussKernel(int radius, float sigma) {
	int sizeOfArray = radius*2+1;
    float* gauss = (float*)malloc(sizeOfArray*sizeof(float));
    float sum = 0.0;
    float val = 0.0;
    for (int i = -radius; i <= radius; i++) {
        val = 1.0/(sqrt(2*PI)*sigma) * exp(pow(i, 2) * -1.0/(2*pow(sigma, 2)));
        // val = exp( (-1.0) * pow(i, 2) / ( 2 * pow(sigma, 2) ) );
        gauss[i+radius] = val;
        sum += val;
    }
    for (int i = 0; i < sizeOfArray; i++ ) {
        gauss[i] /= sum;
    }
    return gauss;
}

float* binomialFilter(int radius) {
    int sizeOfArray = radius*2+1;
    float* binomial = (float*)malloc(sizeOfArray*sizeof(float));
    int sum = 0;
    binomial[0] = 1;
    for (int i = 1; i < sizeOfArray; i++) {
        binomial[i] = 0;
    }
    for (int i = 1; i < sizeOfArray; i++) {
        for (int j = i; j > 0; j--) {
            binomial[j] += binomial[j-1];
        }
    }
    for (int i = 0; i < sizeOfArray; i++) {
        sum += binomial[i];
    }
    for (int i = 0; i < sizeOfArray; i++) {
        binomial[i] /= sum;
    }
    return binomial;
}

float* boxKernel(int radius) {
	int length = radius*2+1;
    float* box = (float*)malloc(length*sizeof(float));
    for (int i = -radius; i <= radius; i++) {
        box[i+radius] = 1.0 / length;
    }
    return box;
}

float* secondDerivative() {
    float* derivative = (float*)malloc(3*sizeof(float));
    derivative[0] = 1;
    derivative[1] = -2;
    derivative[2] = 1;
    return derivative;
}

int* morphologicalStandardFilter(int radius) {
    int size = 2*radius+1;
    int* filter = (int*)malloc(size*sizeof(int));
    for (int i = 0; i < size; i++) {
        filter[i] = 1.0;
    }
    return filter;
}

float** laplacianOfGaussianFilter(int radius, float sigma) {
    int sizeOfArray = 2*radius+1;
    float** laplaceOfGaussFilter = (float**)malloc(sizeOfArray*sizeof(float*));
    for (int i = 0; i < sizeOfArray; i++) {
        laplaceOfGaussFilter[i] = (float*)malloc(sizeOfArray*sizeof(float));
    }
    double sum = 0.0;
    double val = 0.0;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; ++j) {
            // val = (-1) * ( pow(j, 2) + pow(i, 2) - 2.0 * pow(sigma, 2) ) / ( pow(sigma, 4) ) * exp( (-1) * ( pow(j, 2) + pow(i, 2) ) / ( 2.0 * pow(sigma, 2) ) );
            val = (-1.0) / ( PI * pow(sigma, 4) ) * ( 1.0 - ( pow(j, 2) + pow(i, 2) ) / ( 2.0 * pow(sigma, 2) ) )
                  * exp( (-1) * ( pow(j, 2) + pow(i, 2) ) / ( 2 * pow(sigma, 2) ) );
            laplaceOfGaussFilter[i+radius][j+radius] = val;
            sum += val;
        }
    }
    for (int i = 0; i < sizeOfArray; i++ ) {
        for (int j = 0; j < sizeOfArray; j++) {
            laplaceOfGaussFilter[i][j] /= sum;
            // laplaceOfGaussFilter[i][j] *= 255;
        }
    }
    return laplaceOfGaussFilter;
}