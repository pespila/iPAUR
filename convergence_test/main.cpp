#include <stdio.h>
#include <math.h>
#include "analytical_operators.h"
#include "main.h"
#include "ms_minimizer.h"
#include "ms_minimizer_video.h"
#include "image.h"

int main(int argc, const char* argv[]) {
    run(argc, argv);
    return 0;
}

unsigned char* alloc_image_data(int height, int width) {
    return (unsigned char*)malloc(height*width*sizeof(unsigned char));
}

double* alloc_double_array(int height, int width) {
    return (double*)malloc(height*width*sizeof(double));
}

void run(int argc, const char* argv[]) {
    gray_img* gray;
    if (argc == 1) {
        gray = initalize_raw_image(10, 10, 0);
    } else if (argc == 2) {
        gray = read_image_data(argv[1]);
        for (int i = 0; i < gray->image_height; i++)
        {
            for (int j = 0; j < gray->image_width; j++)
            {
                printf("%d ", gray->image_data[j + i * gray->image_width]);
            }
            printf("\n");
        }
    } else {
        gray = read_image_data(argv[1]);

        // for (int i = 0; i < gray->image_height; i++)
        // {
        //     for (int j = 0; j < gray->image_width; j++)
        //     {
        //         int random = rand()%3 + 1;
        //         if (random == 1) {
        //             int randomi = rand()%2 + 1;
        //             if (randomi == 1) {
        //                 gray->approximation[j + i * gray->image_width] = 255;
        //             } else {
        //                 gray->approximation[j + i * gray->image_width] = 0;
        //             }
        //         }
        //     }
        // }

        // tau, lambda, theta, alpha, gamma, delta, spacing
        // TV-L1
        // struct parameter* input_parameter = set_input_parameter(gray, 0.02, 5.0, 1.0, 0.05, 0.0, 0.0, 0); // Algorithm 1: TV-L1 like paper
        // struct parameter* input_parameter = set_input_parameter(gray, 0.35, 0.6, 0.0, 0.0, 0.7*0.6, 0.0, 0); // Algorithm 1: Arrow-Hurwicz TV-L1
        // struct parameter* input_parameter = set_input_parameter(gray, 0.35, 0.6, 0.0, 0.0, 0.7*0.6, 0.0, 0); // Algorithm 2: Arrow-Hurwicz TV-L1
        struct parameter* input_parameter = set_input_parameter(gray, 0.35, 0.7, 1.0, 0.05, 0.0, 0.0, 0); // Algorithm 1: TV-L1 c.f. article.pdf

        // (Huber-) ROF
        // struct parameter* input_parameter = set_input_parameter(gray, 0.05, 4.0, 0.0, 0.0, 0.7*4.0, 0.0, 1); // Algorithm 1: Arrow-Hurwicz ROF
        // struct parameter* input_parameter = set_input_parameter(gray, 0.05, 4.0, 0.0, 0.0, 0.7*4.0, 0.0, 1); // Algorithm 2: Arrow-Hurwicz ROF
        // struct parameter* input_parameter = set_input_parameter(gray, 0.05, 4.0, 0.0, 0.05, 0.0, 0.0, 1); // Algorithm 1: Arrow-Hurwicz Huber-ROF
        // struct parameter* input_parameter = set_input_parameter(gray, 0.01, 8.0, 1.0, 0.0, 0.0, 0.0, 1); // Algorithm 1: ROF model, lambda = 8
        // struct parameter* input_parameter = set_input_parameter(gray, 0.01, 16.0, 1.0, 0.0, 0.0, 0.0, 1); // Algorithm 1: ROF model, lambda = 16
        // struct parameter* input_parameter = set_input_parameter(gray, 0.01, 5.0, 1.0, 0.05, 0.0, 0.0, 1); // Algorithm 1: Huber-ROF model, lambda = 5, alpha = 0,05
        // struct parameter* input_parameter = set_input_parameter(gray, 0.01, 5.0, 1.0, 0.05, 0.7*5.0, 0.0, 1); // Algorithm 2: Huber-ROF model, lambda = 5, alpha = 0,05
        // struct parameter* input_parameter = set_input_parameter(gray, 0.01, 5.0, 1.0, 0.1, 5.0, 0.1, 1); // Algorithm 3: Huber-ROF model, lambda = 5, alpha = 0,05
        
        // primal_dual_algorithm(gray, (&huber_rof_proximation_f_star), (&proximation_g), input_parameter, 1, 0, 1000);
        primal_dual_algorithm(gray, (&huber_rof_proximation_f_star), (&proximation_tv_l1_g), input_parameter, 0, 0, atoi(argv[3]), argv[4]);

        // primal_dual_algorithm_video(gray, (&huber_rof_proximation_f_star), (&proximation_g), input_parameter, 1, 1, 100);
        // primal_dual_algorithm_video(gray, (&huber_rof_proximation_f_star), (&proximation_tv_l1_g), input_parameter, 0, 0, 500);
        
        free(input_parameter);

        write_image_data(gray, argv[2]);
    }
    destroy_image(gray);
}