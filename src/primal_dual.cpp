#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "primal_dual.h"
#include "tv_l1_model.h"
#include "huber_rof_model.h"
#include "util.h"
#include "image.h"

void dykstra_projection(int nRow, int nCol, int max_dykstra_iter, int sigma, float *Src) {
	int n_constr = nRow*(nRow - 1)/2;
	// double *q;
	double dykstra_tol = 1e-04;
	
	// q = mxGetPr(mxCreateDoubleMatrix(n_constr,nCol,mxREAL));
	float* q = (float*)malloc(n_constr*nCol*sizeof(float));
	int i;
	for (i = 1; i<= max_dykstra_iter; i++)
	{
		double error = 0;
		int q_idx = 1;
		for (int i1 = 1; i1<= nRow - 1; i1++)
		{
			double p_i1_x = Src[i1 - 1];
			double p_i1_y = Src[i1 - 1 + nRow];
			for (int i2 = i1 + 1; i2<= nRow; i2++)
			{
				double p_i2_x = Src[i2 - 1];
				double p_i2_y = Src[i2 - 1 + nRow];
				double r_x,r_y;
				r_x = p_i2_x - p_i1_x + q[q_idx - 1];
				r_y = p_i2_y - p_i1_y + q[q_idx - 1 + n_constr];
				double norm = sqrt(r_x*r_x + r_y*r_y);
				if (norm <= sigma)
				{
					r_x = 0;
					r_y = 0;
				}
				else
				{
					r_x = (1 - sigma/norm)*r_x;
					r_y = (1 - sigma/norm)*r_y;
				}
				//make update
				double e_x,e_y;
				e_x = r_x - q[q_idx - 1];
				e_y = r_y - q[q_idx - 1 + n_constr];
				p_i1_x = p_i1_x + 0.5*e_x;
				p_i1_y = p_i1_y + 0.5*e_y;
				p_i2_x = p_i2_x - 0.5*e_x;
				p_i2_y = p_i2_y - 0.5*e_y;
				//write back
				Src[i2 - 1] = p_i2_x;
				Src[i2 - 1 + nRow] = p_i2_y;
				q[q_idx - 1] = r_x;
				q[q_idx - 1 + n_constr] = r_y;
				
				double sqrt_e = sqrt(e_x*e_x + e_y*e_y);
				error = MAX(error,sqrt_e);

				q_idx = q_idx + 1;
			}
			Src[i1 - 1] = p_i1_x;
			Src[i1 - 1 + nRow] = p_i1_y;
		}
		if (error < dykstra_tol)
			break;
	}
	// *pl = MAX(*pl,i);
	free(q);
}

// void dykstra_projection(float* proximated1, float* proximated2, float* y1, float* y2, int size, int maxiter) {
// 	float* y_iter1 = (float*)malloc(size*sizeof(float));
// 	float* y_iter2 = (float*)malloc(size*sizeof(float));
// 	float* p1 = (float*)malloc(size*sizeof(float));
// 	float* p2 = (float*)malloc(size*sizeof(float));
// 	float* q1 = (float*)malloc(size*sizeof(float));
// 	float* q2 = (float*)malloc(size*sizeof(float));
	
// 	float y_tmp1 = 0.0;
// 	float y_tmp2 = 0.0;
// 	float absvalue = 0.0;

// 	for (int k = 0; k < maxiter; k++)
// 	{
// 		for (int i = 0; i < size; i++)
// 		{
// 			y_tmp1 = y1[i] + p1[i];
// 			y_tmp2 = y2[i] + p2[i];
// 			absvalue = sqrtf(y_tmp1 * y_tmp1 + y_tmp2 * y_tmp2);
// 			y_iter1[i] = y_tmp1 / fmax(1.0, absvalue);
// 			y_iter2[i] = y_tmp2 / fmax(1.0, absvalue);
// 			p1[i] = y1[i] + p1[i] - y_iter1[i];
// 			p2[i] = y2[i] + p2[i] - y_iter2[i];
// 			y_tmp1 = y_iter1[i] + q1[i];
// 			y_tmp2 = y_iter2[i] + q2[i];
// 			absvalue = sqrtf(y_tmp1 * y_tmp1 + y_tmp2 * y_tmp2);
// 			proximated1[i] = y_tmp1 / fmax(1.0, absvalue);
// 			proximated2[i] = y_tmp2 / fmax(1.0, absvalue);
// 			q1[i] = y_iter1[i] + q1[i] - proximated1[i];
// 			q2[i] = y_iter2[i] + q2[i] - proximated2[i];
// 		}
// 	}

// 	free(y_iter1);
// 	free(y_iter2);
// 	free(p1);
// 	free(p2);
// 	free(q1);
// 	free(q2);
// }

void truncation_operation(float* x_out, float* x_in, int size) {
	for (int i = 0; i < size; i++)
		x_out[i] = fmin(1.0, fmax(0.0, x_in[i]));
}

void primal_dual(gray_img* src, param* parameter, const char* filename, int steps) {
	const int M = src->image_height;
	const int N = src->image_width;
	const int size = M*N;
	float stopping_criterion = 1e-5;
	int i, k;

	float* image = (float*)malloc(size*sizeof(float));
	float* y1 = (float*)malloc(size*sizeof(float));
	float* y2 = (float*)malloc(size*sizeof(float));
	float* grad_u1 = (float*)malloc(size*sizeof(float));
	float* grad_u2 = (float*)malloc(size*sizeof(float));
	float* proximated1 = (float*)malloc(size*sizeof(float));
	float* proximated2 = (float*)malloc(size*sizeof(float));
	float* x = (float*)malloc(size*sizeof(float));
	float* x_bar = (float*)malloc(size*sizeof(float));
	float* x_current = (float*)malloc(size*sizeof(float));
	float* divergence = (float*)malloc(size*sizeof(float));
	float* energy = (float*)malloc((steps+1)*sizeof(float));

	for (i = 0; i < size; i++) {
		x[i] = src->approximation[i];
		x_bar[i] = src->approximation[i];
		image[i] = src->approximation[i];
		proximated1[i] = 0.0;
		proximated2[i] = 0.0;
	}

	cv::Mat gray_image, writing_image;
	cv::VideoWriter output_cap;

	if (filename != NULL) {
		output_cap = cv::VideoWriter(filename, CV_FOURCC('m', 'p', '4', 'v'), 50, cv::Size(src->image_width, src->image_height), false);
		if (!output_cap.isOpened()) {
			printf("ERROR by opening!\n");
	   	}
	}

	gradient_of_image_value_without_scaling(grad_u1, grad_u2, x_bar, M, N);
	float normalization_value = computed_energy_of_tv_l1_functional(x_bar, grad_u1, grad_u2, x_bar, size);
	energy[0] = 1.0;

	for (k = 1; k <= steps; k++) {
		printf("Step: %d\n", k);
		for (i = 0; i < size; i++) {x_current[i] = x[i];}
		gradient_of_image_value_without_scaling(y1, y2, x_bar, M, N);
		for (i = 0; i < size; i++) {y1[i] = parameter->sigma * y1[i] + y1[i];}
		for (i = 0; i < size; i++) {y2[i] = parameter->sigma * y2[i] + y2[i];}
		// for (i = 0; i < size; i++) {y1[i] = parameter->sigma * y1[i] + proximated1[i];}
		// for (i = 0; i < size; i++) {y2[i] = parameter->sigma * y2[i] + proximated2[i];}
		dykstra_projection(M, N, 200, 8, y1);
		dykstra_projection(M, N, 200, 8, y2);
		// dykstra_projection(proximated1, proximated2, y1, y2, size, 50);
		divergence_of_dual_vector_without_scaling(divergence, y1, y2, M, N);
		// divergence_of_dual_vector_without_scaling(divergence, proximated1, proximated2, M, N);
		for (i = 0; i < size; i++) {divergence[i] = parameter->tau * divergence[i] + x_current[i];}
		truncation_operation(x, divergence, size);
		for (i = 0; i < size; i++) {x_bar[i] = x[i] + parameter->theta * (x[i] - x_current[i]);}

		gradient_of_image_value_without_scaling(grad_u1, grad_u2, x_bar, M, N);
		energy[k] = computed_energy_of_tv_l1_functional(x_bar, grad_u1, grad_u2, x_bar, size) / normalization_value;
		if (k > 200) {
			if (energy[k - 1] - energy[k] < stopping_criterion) {
				for (i = 0; i < size; i++) {src->approximation[i] = (unsigned char)x_bar[i];}
				printf("Steps: %d\n", k);
				break;
			}
		}
		if (filename != NULL) {
			gray_image = convert_into_opencv_image(x_bar, src->image_height, src->image_width, src->image_type);
			cvtColor(gray_image, writing_image, CV_GRAY2BGR);
			output_cap.write(writing_image);
		}
		if (k == steps) {
			for (i = 0; i < size; i++) {src->approximation[i] = (unsigned char)x_bar[i];}
			printf("Steps: %d\n", k);
			break;
		}
	}
	energy_to_file(energy, steps+1, "./plot.dat");

	free(y1);
	free(y2);
	free(proximated1);
	free(proximated2);
	free(x);
	free(image);
	free(x_bar);
	free(x_current);
	free(divergence);
}