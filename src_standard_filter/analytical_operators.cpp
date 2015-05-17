#include "analytical_operators.h"

Analytical_Operators::Analytical_Operators(int height, int width, double tau, double lambda, double theta, double alpha) {
    this->height = height;
    this->width = width;
    dual_x1 = (double*)malloc(height*width*sizeof(double));
    dual_x2 = (double*)malloc(height*width*sizeof(double));
    dual_proximated1 = (double*)malloc(height*width*sizeof(double));
    dual_proximated2 = (double*)malloc(height*width*sizeof(double));
    this->tau = tau;
    this->lambda = lambda;
    this->theta = theta;
    this->alpha = alpha;
    this->gamma = 0.0;
    this->L2 = 8.0 * height * width;
    this->sigma = 1.0 / (tau * this->L2);
}

Analytical_Operators::~Analytical_Operators() {
    free(dual_x1);
    free(dual_x2);
    free(dual_proximated1);
    free(dual_proximated2);
}

void Analytical_Operators::update_theta_tau_sigma(void) {
    this->theta = 1.0 / ( sqrt( 1.0 + 2.0 * this->gamma * this->tau ) );
    this->tau = this->theta * this->tau;
    this->sigma = this->sigma / this->theta;
}

void Analytical_Operators::add_vectors(double parameter_x_in1, double parameter_x_in2, double* x_in1, double* x_in2, double* x_out) {
    const int M = this->height;
    const int N = this->width;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            x_out[j + i * N] = parameter_x_in1 * x_in1[j + i * N] + parameter_x_in2 * x_in2[j + i * N];
        }
    }
}

void Analytical_Operators::add_dual_variables(double parameter_x_in1, double parameter_x_in2) {
    const int M = this->height;
    const int N = this->width;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            dual_x1[j + i * N] = parameter_x_in1 * dual_proximated1[j + i * N] + parameter_x_in2 * dual_x1[j + i * N];
            dual_x2[j + i * N] = parameter_x_in1 * dual_proximated2[j + i * N] + parameter_x_in2 * dual_x2[j + i * N];
        }
    }
}

void Analytical_Operators::gradient_of_image_value(double gradient_scalar, double* image) {
    const int M = this->height;
    const int N = this->width;
    const double H_HEIGHT = 1.0/M;
    const double H_WIDTH = 1.0/N;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            dual_x1[j + i * N] = i + 1 < M ? gradient_scalar * (image[j + (i+1) * N] - image[j + i * N]) / H_HEIGHT : 0.0;
            dual_x2[j + i * N] = j + 1 < N ? gradient_scalar * (image[j + 1 + i * N] - image[j + i * N]) / H_WIDTH : 0.0;
        }
    }
}

void Analytical_Operators::divergence_of_dual_vector(double divergence_scalar, double* divergence) {
    const int M = this->height;
    const int N = this->width;
    const double H_HEIGHT = 1.0/M;
    const double H_WIDTH = 1.0/N;
    double computed_argument_x1 = 0.0;
    double computed_argument_x2 = 0.0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0) computed_argument_x1 = dual_proximated1[j + i * N];
            else if (i == M-1) computed_argument_x1 = -dual_proximated1[j + (i-1) * N];
            else computed_argument_x1 = dual_proximated1[j + i * N] - dual_proximated1[j + (i-1) * N];

            if (j == 0) computed_argument_x2 = dual_proximated2[j + i * N];
            else if (j == N-1) computed_argument_x2 = -dual_proximated2[(j-1) + i * N];
            else computed_argument_x2 = dual_proximated2[j + i * N] - dual_proximated2[(j-1) + i * N];

            divergence[j + i * N] = divergence_scalar * (computed_argument_x1 / H_HEIGHT + computed_argument_x2 / H_WIDTH);
        }
    }
}