#include "image.h"
#include "primal_dual.h"

void run(int, const char*[]);

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }

    run(argc, argv);

    return 0;
}

void run(int argc, const char* argv[]) {
    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();
    if (atoi(argv[3]) == 1) {
        gray_img* gray;
        gray = read_image_data(argv[1]);
        if (atoi(argv[4]) == 1) {
            param* parameter = set_input_parameter(0.1, 5.0);
            primal_dual(gray, parameter, 1, 200);
        } else {
            printf("Too few arguments!\n");
        }
        write_image_data(gray, argv[2]);
        destroy_image(gray);
    } else {
        printf("Too few arguments!\n");
    }
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
}