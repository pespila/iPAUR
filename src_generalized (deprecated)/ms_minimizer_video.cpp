#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
// #include "opencv/cv.h"
// #include "opencv/highgui.h"
#include "analytical_operators.h"
#include "ms_minimizer.h"
#include "ms_minimizer_video.h"
#include "main.h"
#include "image.h"

using namespace cv;

Mat write_image_data(gray_img* image) {
    Mat img(image->image_height, image->image_width, image->image_type);
    for (int i = 0; i < image->image_height; i++) {
        for (int j = 0; j < image->image_width; j++) {
            img.at<uchar>(i, j) = image->approximation[j + i * image->image_width];
        }
    }
    return img;
}

void primal_dual_algorithm_video(gray_img* src, void (*prox_f)(struct dual_vector_2d, struct dual_vector_2d, struct parameter*, int, int), void (*prox_g)(double*, double*, gray_img*, struct parameter*), struct parameter* input_parameter, int spacing, int update, int steps) {
	const int M = src->image_height;
	const int N = src->image_width;

	struct dual_vector_2d y = {alloc_double_array(M, N), alloc_double_array(M, N)};
	struct dual_vector_2d proximated = {alloc_double_array(M, N), alloc_double_array(M, N)};
	double* x = alloc_double_array(M, N);
	double* x_bar = alloc_double_array(M, N);
	double* x_current = alloc_double_array(M, N);
	double* divergence = alloc_double_array(M, N);

	init_vectors(x, x_bar, proximated, src, M, N);

	Mat gray, color;

	VideoWriter output_cap("/Users/michael/Documents/Programming/image-processing/img/eye.mp4",
                                  CV_FOURCC('m', 'p', '4', 'v'),
                                  50,
                                  cv::Size(src->image_width, src->image_height),
                                  false);
	if (!output_cap.isOpened()) {
		printf("ERROR by opening!\n");
       // return -1;
   	}

	for (int k = 1; k <= steps; k++) {
		add_vectors(x_current, x, x, 1.0, 0.0, M, N);
		gradient_of_image_value(y, input_parameter->sigma, x_bar, M, N, spacing);
		add_vectors(y.x1, proximated.x1, y.x1, 1.0, 1.0, M, N);
		add_vectors(y.x2, proximated.x2, y.x2, 1.0, 1.0, M, N);
		prox_f(proximated, y, input_parameter, M, N);
		divergence_of_dual_vector(divergence, input_parameter->tau, proximated, M, N, spacing);
		add_vectors(divergence, x_current, divergence, 1.0, 1.0, M, N);
		prox_g(x, divergence, src, input_parameter);
		if (update) {
			update_input_parameters(input_parameter);
		}
		add_vectors(x_bar, x_current, x, -input_parameter->theta, (1 + input_parameter->theta), M, N);
		set_approximation(src, x_bar);
		gray = write_image_data(src);
		cvtColor(gray, color, CV_GRAY2BGR);
		output_cap.write(color);
		if (k == steps) {
			set_approximation(src, x_bar);
		}
	}
	// waitKey();

	free_memory_of_vectors(x, x_bar, x_current, divergence, y, proximated);
}