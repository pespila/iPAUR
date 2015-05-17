#include "rof_model.h"

Huber_ROF_Model::Huber_ROF_Model(int height, int width, double tau, double lambda, double theta, double alpha) : Analytical_Operators(height, width, tau, lambda, theta, alpha) {
    x = (double*)malloc(height*width*sizeof(double));
    x_bar = (double*)malloc(height*width*sizeof(double));
    x_current = (double*)malloc(height*width*sizeof(double));
    divergence = (double*)malloc(height*width*sizeof(double));
}

Huber_ROF_Model::~Huber_ROF_Model() {
    free(x);
    free(x_bar);
    free(x_current);
    free(divergence);
}

void Huber_ROF_Model::init_vectors(GrayscaleImage& src) {
    for (int i = 0; i < this->height*this->width; i++) {
        this->x[i] = src.get_gray_pixel_at_position(0, i);
        this->x_bar[i] = this->x[i];
        this->dual_proximated1[i] = 0.0;
        this->dual_proximated2[i] = 0.0;
    }
}

void Huber_ROF_Model::proximation_g(GrayscaleImage& src, double* x_in, double* x_out) {
    const int M = this->height;
    const int N = this->width;
    double tau_mult_lambda = this->tau * this->lambda;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            x_out[j + i * N]  = (x_in[j + i * N] + tau_mult_lambda * src.get_gray_pixel_at_position(i, j)) / (1.0 + tau_mult_lambda);
        }
    }
}

void Huber_ROF_Model::proximation_f_star() {
    const int M = this->height;
    const int N = this->width;
    double vector_norm = 0.0;
    double sigma_mult_alpha = this->sigma * this->alpha;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            vector_norm = sqrt(dual_x1[j + i * N] * dual_x1[j + i * N] + dual_x2[j + i * N] * dual_x2[j + i * N]);
            dual_proximated1[j + i * N] = (dual_x1[j + i * N]/(1.0 + sigma_mult_alpha))/fmax(1.0, vector_norm/(1.0 + sigma_mult_alpha));
            dual_proximated2[j + i * N] = (dual_x2[j + i * N]/(1.0 + sigma_mult_alpha))/fmax(1.0, vector_norm/(1.0 + sigma_mult_alpha));
        }
    }
}

void Huber_ROF_Model::set_approximation(GrayscaleImage& dst) {
    const int M = this->height;
    const int N = this->width;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            dst.set_gray_pixel_at_position(i, j, (unsigned char)this->x_bar[j + i * N]);
        }
    }
}

void Huber_ROF_Model::primal_dual_algorithm(GrayscaleImage& src, GrayscaleImage& dst) {
    dst.reset_image(src.get_height(), src.get_width(), src.get_type());
    init_vectors(src);

    for (int k = 1; k <= 1000; k++) {
        add_vectors(1.0, 0.0, this->x, this->x, this->x_current);
        gradient_of_image_value(this->sigma, this->x_bar);
        add_dual_variables(1.0, 1.0);
        proximation_f_star();
        divergence_of_dual_vector(this->tau, this->divergence);
        add_vectors(1.0, 1.0, this->divergence, this->x_current, this->divergence);
        proximation_g(src, this->divergence, this->x);
        add_vectors(-this->theta, (1.0 + this->theta), this->x_current, this->x, this->x_bar);
        if (k == 1000) {
            set_approximation(dst);
        }
    }
}