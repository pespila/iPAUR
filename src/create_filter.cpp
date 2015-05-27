#include "create_filter.h"

float* gaussian_kernel(int radius, float sigma) {
    int kernel_size = radius * 2 + 1;
    float* gauss_kernel = (float*)malloc(kernel_size*sizeof(float));
    float value = 0.0;
    float sum = 0.0;
    int i;
    for (i = (-1) * radius; i <= radius; i++) {
        value = 1.0 / (sqrt(2 * PI) * sigma) * exp( pow(i, 2) * (-1.0) / (2.0 * pow(sigma, 2)) );
        gauss_kernel[i + radius] = value;
        sum += value;
    }
    for (i = 0; i < kernel_size; i++) {
        gauss_kernel[i] /= sum;
    }
    return gauss_kernel;
}

float* box_kernel(int radius) {
    int kernel_size = radius * 2 + 1;
    float* kernel = (float*)malloc(kernel_size*sizeof(float));
    for (int i = (-1) * radius; i <= radius; i++) {
        kernel[i + radius] = 1.0 / kernel_size;
    }
    return kernel;
}

float* binomial_filter(int radius) {
    int kernel_size = radius * 2 + 1;
    int sum = 0; 
    int i, j;
    float* binomial = (float*)malloc(kernel_size*sizeof(float));
    binomial[0] = 1.0;
    for (i = 1; i < kernel_size; i++) {
        binomial[i] = 0.0;
    }
    for (i = 1; i < kernel_size; i++) {
        for (j = i; j > 0; j--) {
            binomial[j] += binomial[j - 1];
        }
    }
    for (i = 0; i < kernel_size; i++) {
        sum += binomial[i];
    }
    for (i = 0; i < kernel_size; i++) {
        binomial[i] /= sum;
    }
    return binomial;
}

float* second_derivative() {
    float* derivative = (float*)malloc(3*sizeof(float));
    derivative[0] = 1.0; derivative[1] = -2.0; derivative[2] = 1.0;
    return derivative;
}

int* morphological_standard_filter(int radius) {
    int size = 2 * radius + 1;
    int* filter = (int*)malloc(size*sizeof(int));
    for (int i = 0; i < size; i++) filter[i] = 1.0;
    return filter;
}

float** laplace_of_gauss_filter(int radius, float sigma) {
    int kernel_size = 2 * radius + 1;
    int i, j;

    float** l_o_g = (float**)malloc(kernel_size*sizeof(float*));
    for (i = 0; i < kernel_size; i++) l_o_g[i] = (float*)malloc(kernel_size*sizeof(float));
    
    double sum = 0.0;
    double value = 0.0;
    for (i = (-1) * radius; i <= radius; i++) {
        for (j = (-1) * radius; j <= radius; j++) {
            value = (-1.0) / ( PI * pow(sigma, 4) ) * ( 1.0 - ( pow(j, 2) + pow(i, 2) ) / ( 2.0 * pow(sigma, 2) ) )
                  * exp( (-1) * ( pow(j, 2) + pow(i, 2) ) / ( 2 * pow(sigma, 2) ) );
            l_o_g[i+radius][j+radius] = value;
            sum += value;
        }
    }
    for (i = 0; i < kernel_size; i++) {
        for (j = 0; j < kernel_size; j++) {
            l_o_g[i][j] /= sum;
        }
    }
    return l_o_g;
}