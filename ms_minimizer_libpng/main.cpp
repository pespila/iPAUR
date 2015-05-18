#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "util.h"
#include "image.h"
#include "huber_rof_model.h"
#include "tv_l1_model.h"

void run(int, const char**);

int main(int argc, const char* argv[]) {
    run(argc, argv);
    return 0;
}

void run(int argc, const char* argv[]) {
    gray_img* gray;
    gray = read_image_data(argv[1]);

    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock(); // start stop-watch
    if (atoi(argv[3]) == 1) {
        param* parameter = set_input_parameter(0.35, 0.7, 1.0, 0.05, 1); // Algorithm 1: TV-L1 c.f. article.pdf
        tv_l1_model(gray, parameter, 1000);
        write_image_data(gray, argv[2]);
    } else if (atoi(argv[3]) == 2) {
        param* parameter = set_input_parameter(0.01, 8.0, 1.0, 0.05, gray->image_height * gray->image_width);
        huber_rof_model(gray, parameter, 1000);
        write_image_data(gray, argv[2]);
    }
    float stop_watch = clock(); // stop stop-watch
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);

    destroy_image(gray);
}