#include <stdlib.h>
#include <stdio.h>
#include <time.h>
// #include <omp.h>
#include "ms_minimizer.h"
#include "image.h"

void run(int, const char**);

int main(int argc, const char* argv[]) {
    run(argc, argv);
    return 0;
}

void run(int argc, const char* argv[]) {
    gray_img* gray;
    gray = read_image_data(argv[1]);
    struct parameter* input_parameter = set_input_parameter(gray, 0.35, 0.7, 1.0, 0.05, 0.0, 0.0, 0); // Algorithm 1: TV-L1 c.f. article.pdf
    // struct parameter* input_parameter = set_input_parameter(gray, 0.01, 8.0, 1.0, 0.0, 0.0, 0.0, 1); // Algorithm 1: ROF model, lambda = 8
    // struct parameter* input_parameter = set_input_parameter(gray, 0.01, 16.0, 1.0, 0.0, 0.0, 0.0, 1); // Algorithm 1: ROF model, lambda = 16
    
    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock(); // start stop-watch
    primal_dual_algorithm(gray, (&huber_rof_proximation_f_star), (&proximation_tv_l1_g), input_parameter, 0, 0, 500);
    // primal_dual_algorithm(gray, (&huber_rof_proximation_f_star), (&proximation_g), input_parameter, 1, 0, 100);
    float stop_watch = clock(); // stop stop-watch
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
    free(input_parameter);

    write_image_data(gray, argv[2]);
    destroy_image(gray);
}