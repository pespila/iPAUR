#include "ms_minimizer.h"

void primal_dual_algorithm(GrayscaleImage& src, GrayscaleImage& dst) {
	int M = src.get_height(), N = src.get_width();
	// double alpha = 10e10;
	double h_M = pow((double)M, -1);
	double h_N = pow((double)N, -1);
	double tau = 0.01;
	// double d = 2.0;
	// double tau = 1.0/(2.0*d);
	// double L = 8.0/(h_M * h_N);
	double sigma = 0.1;
	// double sigma = 1.0/(L*L*tau);
	double lambda = 8.0;
	// double lambda = 64.0;
	double theta = 1.0;

	dst.reset_image(M, N, src.get_type());
	struct dual* y = (struct dual*)malloc(M*N*sizeof(struct dual));
	double* x = (double*)malloc(M*N*sizeof(double));
	double* x_bar = (double*)malloc(M*N*sizeof(double));
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			// x[j + i * N] = sqrt(src.get_gray_pixel_at_position(i, j));
			x[j + i * N] = ((double)src.get_gray_pixel_at_position(i, j)/255.0);
			x_bar[j + i * N] = x[j + i * N];
			y[j + i * N].y1 = 0.0;
			y[j + i * N].y2 = 0.0;
		}
	}

	// int steps = 1;
	// for (int k = 1; k <= steps; k++)
	// {
	// 	for (int i = 0; i < M; i++)
	// 	{
	// 		for (int j = 0; j < N; j++)
	// 		{
	// 			double x0 = x_bar[j + i * N];
	// 			double nablaX_1 = i + 1 < M ? x_bar[j + (i+1) * N] : 0.0;
	// 			double nablaX_2 = j + 1 < N ? x_bar[j + 1 + i * N] : 0.0;
	// 			double y1 = i + 1 < M ? y[j + i * N].y1 + sigma * (nablaX_1 - x0) : 0.0;
	// 			double y2 = j + 1 < N ? y[j + i * N].y2 + sigma * (nablaX_2 - x0) : 0.0;
	// 			y[j + i * N].y1 = sqrt(pow(y1, 2) + pow(y2, 2)) <= sqrt(lambda/alpha * sigma * (sigma + 2.0*alpha)) ? (2.0*alpha)/(sigma + 2.0*alpha) * y1 : 0.0;
	// 			y[j + i * N].y2 = sqrt(pow(y1, 2) + pow(y2, 2)) <= sqrt(lambda/alpha * sigma * (sigma + 2.0*alpha)) ? (2.0*alpha)/(sigma + 2.0*alpha) * y2 : 0.0;


	// 			double y1_0 = i + 1 < M ? y[j + i * N].y1 : 0.0;
	// 			double y1_x = i > 0 ? y[j + (i-1) * N].y1 : 0.0;
	// 			double y2_0 = j + 1 < N ? y[j + i * N].y2 : 0.0;
	// 			double y2_y = j > 0 ? y[j - 1 + i * N].y2 : 0.0;
	// 			// double divergenceY = y1_0 - y1_x + y2_0 - y2_y;
	// 			double divergenceY = y1_x - y1_0 + y2_y - y2_0;
	// 			// double divergenceY = (y1_x - y1_0)/h_M + (y2_y - y2_0)/h_N;


	// 			double x_n = x[j + i * N];
	// 			double x_tmp = x_n + tau * divergenceY;
	// 			x[j + i * N] = (x_tmp + tau * 2.0 * src.get_gray_pixel_at_position(i, j))/(1.0 + tau * 2.0);
	// 			// double tmp = tau * lambda;
	// 			// double difference = x_tmp - src.get_gray_pixel_at_position(i, j);
	// 			// if (difference > tmp) {
	// 			// 	x[j + i * N] = src.get_gray_pixel_at_position(i, j) - tmp;
	// 			// } else if (difference < (-1)*tmp) {
	// 			// 	x[j + i * N] = src.get_gray_pixel_at_position(i, j) + tmp;
	// 			// } else if (abs(difference) <= tmp) {
	// 			// 	cout << "IN" << endl;
	// 			// 	x[j + i * N] = src.get_gray_pixel_at_position(i, j);
	// 			// }
	// 			theta = 1.0/(sqrt(1.0 + 4.0*tau));
	// 			tau = theta * tau;
	// 			sigma = sigma/theta;
	// 			x_bar[j + i * N] = x[j + i * N] + theta * (x[j + i * N] - x_n);
	// 			// x_bar[j + i * N] = 2.0 * x[j + i * N] - x_n;
	// 			cout << x[j + i * N] << "   " << x_bar[j + i * N] << endl;
	// 			if (k == steps) {
	// 				dst.set_gray_pixel_at_position(i, j, (int)x_bar[j + i * N]);
	// 				// cout << (int)dst.get_gray_pixel_at_position(i, j) << " ";
	// 			}
	// 		}
	// 		// cout << endl;
	// 	}
	// }
	double y1 = 0.0, y2 = 0.0, y1_1 = 0.0, y1_2 = 0.0, divergenceY = 0.0, x_tilde = 0.0, x_n_old = 0.0;
	int steps = 25000;
	for (int k = 1; k <= steps; k++)
	{
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				y1 = i + 1 < M ? y[j + i * N].y1 + sigma * (x_bar[j + (i+1) * N] - x_bar[j + i * N])/h_M : 0.0;
				y2 = j + 1 < N ? y[j + i * N].y2 + sigma * (x_bar[j + 1 + i * N] - x_bar[j + i * N])/h_N : 0.0;

				y[j + i * N].y1 = y1/std::max(1.0, sqrt(pow(y1, 2) + pow(y2, 2)));
				y[j + i * N].y2 = y2/std::max(1.0, sqrt(pow(y1, 2) + pow(y2, 2)));
			}
		}
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				y1_1 = i > 0 ? (y[j + i * N].y1 - y[j + (i-1) * N].y1)/h_M : 0.0;
				y1_2 = j > 0 ? (y[j + i * N].y2 - y[j - 1 + i * N].y2)/h_N : 0.0;
				divergenceY = tau * (y1_1 + y1_2);
				x_n_old = x[j + i * N];
				x_tilde = x_n_old + divergenceY;
				x[j + i * N]  = (x_tilde + lambda * tau * (double)src.get_gray_pixel_at_position(i, j))/(1.0 + lambda * tau);
				x_bar[j + i * N] = x[j + i * N] + theta * (x[j + i * N] - x_n_old);
				if (k == steps) {
					// dst.set_gray_pixel_at_position(i, j, pow(x_bar[j + i * N], 2));
					dst.set_gray_pixel_at_position(i, j, (unsigned char)(x_bar[j + i * N]));
				}
			}
		}
	}
	// double y1 = 0.0, y2 = 0.0, y1_1 = 0.0, y1_2 = 0.0, divergenceY = 0.0, x_tilde = 0.0, x_n_old = 0.0;
	// int steps = 1;
	// for (int k = 1; k <= steps; k++)
	// {
	// 	for (int i = 0; i < M; i++)
	// 	{
	// 		for (int j = 0; j < N; j++)
	// 		{
	// 			y1 = i + 1 < M ? y[j + i * N].y1 + sigma * (x_bar[j + (i+1) * N] - x_bar[j + i * N]) : 0.0;
	// 			y2 = j + 1 < N ? y[j + i * N].y2 + sigma * (x_bar[j + 1 + i * N] - x_bar[j + i * N]) : 0.0;

	// 			y[j + i * N].y1 = y1/std::max(1.0, sqrt(pow(y1, 2) + pow(y2, 2)));
	// 			y[j + i * N].y2 = y2/std::max(1.0, sqrt(pow(y1, 2) + pow(y2, 2)));

	// 			y1_1 = i > 0 ? (y[j + i * N].y1 - y[j + (i-1) * N].y1) : 0.0;
	// 			y1_2 = j > 0 ? (y[j + i * N].y2 - y[j - 1 + i * N].y2) : 0.0;
	// 			divergenceY = tau * (y1_1 + y1_2);
	// 			x_n_old = x[j + i * N];
	// 			x_tilde = x_n_old + divergenceY;
	// 			x[j + i * N]  = (x_tilde + lambda * tau * (double)src.get_gray_pixel_at_position(i, j))/(1.0 + lambda * tau);
	// 			x_bar[j + i * N] = x[j + i * N] + theta * (x[j + i * N] - x_n_old);
	// 			if (k == steps) {
	// 				dst.set_gray_pixel_at_position(i, j, (int)x_bar[j + i * N]);
	// 			}
	// 		}
	// 		// cout << endl;
	// 	}
	// }
	// for (int k = 1; k <= steps; k++)
	// {
	// 	for (int i = 0; i < M; i++)
	// 	{
	// 		for (int j = 0; j < N; j++)
	// 		{
	// 			double x0 = x_bar[j + i * N];
	// 			double nablaX_1 = i + 1 < M ? x_bar[j + (i+1) * N] : 0.0;
	// 			double nablaX_2 = j + 1 < N ? x_bar[j + 1 + i * N] : 0.0;
	// 			double y1 = i + 1 < M ? sigma * nablaX_1 - x0 : 0.0;
	// 			double y2 = j + 1 < N ? sigma * nablaX_2 - x0 : 0.0;
	// 			y[j + i * N].y1 = y1/std::max(1.0, sqrt(pow(y1, 2) + pow(y2, 2)));
	// 			y[j + i * N].y2 = y2/std::max(1.0, sqrt(pow(y1, 2) + pow(y2, 2)));


	// 			double y1_0 = i + 1 < M ? y[j + i * N].y1 : 0.0;
	// 			double y1_x = i > 0 ? y[j + (i-1) * N].y1 : 0.0;
	// 			double y2_0 = j + 1 < N ? y[j + i * N].y2 : 0.0;
	// 			double y2_y = j > 0 ? y[j - 1 + i * N].y2 : 0.0;
	// 			// double divergenceY = y1_0 - y1_x + y2_0 - y2_y;
	// 			double divergenceY = y1_x - y1_0 + y2_y - y2_0;
	// 			// double divergenceY = (y1_x - y1_0)/h_M + (y2_y - y2_0)/h_N;


	// 			double x_n = x[j + i * N];
	// 			double x_tmp = x_n + tau * divergenceY;
	// 			// x[j + i * N] = (x_tmp + tau * lambda * src.get_gray_pixel_at_position(i, j))/(1.0 + tau * lambda);
	// 			double tmp = tau * lambda;
	// 			double difference = x_tmp - src.get_gray_pixel_at_position(i, j);
	// 			if (difference > tmp) {
	// 				x[j + i * N] = src.get_gray_pixel_at_position(i, j) - tmp;
	// 			} else if (difference < (-1)*tmp) {
	// 				x[j + i * N] = src.get_gray_pixel_at_position(i, j) + tmp;
	// 			} else if (abs(difference) <= tmp) {
	// 				cout << "IN" << endl;
	// 				x[j + i * N] = src.get_gray_pixel_at_position(i, j);
	// 			}
	// 			x_bar[j + i * N] = 2.0 * x[j + i * N] - x_n;
	// 			cout << x[j + i * N] << "   " << x_bar[j + i * N] << endl;
	// 			if (k == steps) {
	// 				dst.set_gray_pixel_at_position(i, j, (int)x_bar[j + i * N]);
	// 				// cout << (int)dst.get_gray_pixel_at_position(i, j) << " ";
	// 			}
	// 		}
	// 		// cout << endl;
	// 	}
	// }
	// for (int k = 1; k <= steps; k++)
	// {
	// 	for (int i = 0; i < M; i++)
	// 	{
	// 		for (int j = 0; j < N; j++)
	// 		{
	// 			double y1 = i+1 == M ? y[j + i * N].y1 : y[j + i * N].y1 + sigma * (x_bar[j + (i+1) * N] - x_bar[j + i * N])/h_M;
	// 			double y2 = j+1 == N ? y[j + i * N].y2 : y[j + i * N].y2 + sigma * (x_bar[j + 1 + i * N] - x_bar[j + i * N])/h_N;
	// 			// cout << x[j + i * N] << endl;
	// 			// cout << y1 << "   " << y2 << endl;
	// 			y[j + i * N].y1 = y1/std::max(1.0, abs(y1));
	// 			y[j + i * N].y2 = y2/std::max(1.0, abs(y2));
	// 			// cout << y[j + i * N].y1 << "   " << y[j + i * N].y2 << endl;
	// 			double y1_tmp = i > 0 ? y[j + (i-1) * N].y1 : 0.0;
	// 			double y2_tmp = j > 0 ? y[j - 1 + i * N].y2 : 0.0;
	// 			y1 = i+1 == M ? 0.0 : (y1_tmp - y[j + i * N].y1)/h_M;
	// 			y2 = j+1 == N ? 0.0 : (y2_tmp - y[j + i * N].y2)/h_N;
	// 			// cout << y1 << "   " << y2 << endl;
	// 			double x_n = x[j + i * N];
	// 			// cout << x_n << endl;
	// 			double x_tmp = x_n + tau * (y1 + y2);
	// 			// cout << x_tmp << endl;
	// 			x[j + i * N] = (x_tmp + tau * lambda * src.get_gray_pixel_at_position(i, j))/(1 + tau * lambda);
	// 			// cout << x[j + i * N] << endl;
	// 			x_bar[j + i * N] = 2 * x[j + i * N] - x_n;
	// 			// cout << x_bar[j + i * N] << endl;
	// 			if (k == steps) {
	// 				dst.set_gray_pixel_at_position(i, j, (int)x_bar[j + i * N]);
	// 			}
	// 		}
	// 	}
	// }
}

// struct vector_2D discrete_nabla(u, i) {
// 	int M = u.get_height(), N = u.get_width();
// 	double h_M = pow((double)M, -1);
// 	double h_N = pow((double)N, -1);

// }

// struct dual updateDual(p, x_bar, sigma, h_M, h_N, i) {
// 	struct dual dual_sum;
// 	dual_sum.x = p[i].x + sigma * (x_bar[i+M] - x_bar[i]) / h_M;
// 	dual_sum.y = p[i].y + sigma * (x_bar[i+1] - x_bar[i]) / h_N;
// 	return dual_sum;
// }

// double updatePrimal(x, p, tau, h_M, h_N, i) {
// 	double primal_sum = x[i-1] - tau * ((p[i+M].x - p[i].x) / h_M + (p[i+1].y - p[i].y) / h_N);
// 	return primal_sum;
// }

// struct dual prox_dual(struct dual dual_pair) {
// 	dual_pair.x = dual_pair.x/std::max(1.0, abs(dual_pair.x));
// 	dual_pair.y = dual_pair.y/std::max(1.0, abs(dual_pair.y));
// 	return dual_pair;
// }

// void prox_primal(u, u_bar, f, lambda, i) {
// 	double numerator = u_bar[i] + tau * lambda * f[i];
// 	double denominator = 1 + tau * lambda;
// 	u[i] = numerator / denominator;
// }

// void dual_ascent(p, x_bar, sigma, i) {
// 	struct dual dual_pair = updateDual(p, x_bar, sigma, h_N, h_M);
// 	dual_pair = prox_dual(dual_pair);
// 	p[i].x = dual_pair.x;
// 	p[i].y = dual_pair.y;
// }

// void primal_descent(f, u, x, p, tau, lambda, i) {
// 	double primal_value = updatePrimal(x, p, tau, h_N, h_M);
// 	prox_primal(u, primal_value, f, lambda);
// }

// void rof(f, u, x, p, x_bar, iterations) {
// 	int M = f.get_height(), N = f.get_width();
// 	double tau = 0.01;
// 	double L = 8.0/pow(h_M * h_N, 2);
// 	double sigma = 1.0/(L*L*tau);
// 	double lambda = 8.0;
// 	double h_M = 1.0/(double)M;
// 	double h_N = 1.0/(double)N;
// 	for (int k = 1; k < iterations; k++)
// 	{
// 		for (int i = 0; i < M*N; i++)
// 		{
// 			dual_ascent(p, x_bar, sigma, i);
// 			primal_descent(f, u, x, p, tau, lambda, i);
// 			x_bar[i] = 2*x[i] - x[i-1];
// 		}
// 	}
// }

// void init(float* u, float* p, struct dual* d, GrayscaleImage& f) {
//     int height = f.get_height(), width = f.get_width();
//     for (int i = 0; i < height; i++)
//     {
//         for (int j = 0; j < width; j++)
//         {
//             u[j + i * width] = f.get_gray_pixel_at_position(i, j)/255.0;
//             p[j + i * width] = f.get_gray_pixel_at_position(i, j)/255.0;
//             d[j + i * width].x = 0.0;
//             d[j + i * width].y = 0.0;
//         }
//     }
// }

// float project_x(float x, float u, float tau, float lambda) {
// 	return (std::min((float)1.0, std::max((float)0.0, x))); 
// }

// float project_y(float y) {
// 	return (y/(std::max((float)1.0, abs(y))));
// }

// void primal_descent(float* p, struct dual* d, float sigma, int height, int width) {
// 	float x0 = 0.0, y0 = 0.0, u0 = 0.0, grad_x = 0.0, grad_y = 0.0;
// 	for (int i = 0; i < height; i++) {
// 	    for (int j = 0; j < width; j++) {
// 	        x0 = i+1 == height ? 0.0 : p[j + (i+1) * width];
// 	        y0 = j+1 == width ? 0.0 : p[j + 1 + i * width];
// 	        u0 = p[j + i * width];
// 	        grad_x = (x0 - u0) * sigma;
// 	        grad_y = (y0 - u0) * sigma;
// 	        d[j + i*width].x += grad_x;
// 	        d[j + i*width].y += grad_y;
// 	        d[j + i*width].x = project_y(d[j + i * width].x);
// 	        d[j + i*width].y = project_y(d[j + i * width].y);
// 	    }
// 	}
// }

// void dual_ascent(float* u, float* p, struct dual* d, float tau, float lambda, int height, int width) {
// 	float x0 = 0.0, y0 = 0.0, u0_x = 0.0, u0_y = 0.0, div_x = 0.0, div_y = 0.0, u_cur = 0.0;
// 	for (int i = 0; i < height; i++) {
// 	    for (int j = 0; j < width; j++) {
// 	        x0 = i+1 == height ? 0.0 : d[j + (i+1) * width].x;
// 	        y0 = j+1 == width ? 0.0 : d[j + 1 + i * width].y;
// 	        u0_x = d[j + i * width].x;
// 	        u0_y = d[j + i * width].y;
// 	        div_x = (x0 - u0_x) * tau;
// 	        div_y = (y0 - u0_y) * tau;
// 	        u_cur = p[j + i*width];
// 	        p[j + i * width] -= (div_x + div_y);
// 	        p[j + i * width] = project_x(p[j + i * width], (float)(pow(i, 2)/height+pow(j, 2)/width), tau, lambda);
// 	        u[j + i * width] = 2.0 * p[j + i * width] - u_cur;
// 	    }
// 	}
// }

// void primal_dual_algorithm(GrayscaleImage& src, GrayscaleImage& dst, int runs) {
// 	int height = src.get_height(), width = src.get_width();
// 	dst.reset_image(height, width, src.get_type());
// 	float* u = (float*)malloc(height*width*sizeof(float));
// 	float* p = (float*)malloc(height*width*sizeof(float));
// 	struct dual* d = (struct dual*)malloc(height*width*sizeof(struct dual));
// 	float L = 1.0/sqrt(12), tau = 0.02, sigma = 1.0/(L*tau), lambda = 0.1;
// 	init(u, p, d, src);

// 	for (int i = 0; i < runs; i++) {
// 		primal_descent(p, d, sigma, height, width);
// 		dual_ascent(u, p, d, tau, lambda, height, width);
// 	}

// 	for (int i = 0; i < height; i++) {
//         for (int j = 0; j < width; j++) {
//             dst.set_gray_pixel_at_position(i, j, (short)(255.0*p[j + i * width]));
//         }
//     }
// }