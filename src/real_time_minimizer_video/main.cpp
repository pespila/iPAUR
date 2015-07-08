#include "image.h"
#include "rgb.h"
#include "grayscale.h"
#include "parameter.h"
#include "real_time_minimizer.h"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }

    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();
    RGBImage in, out;
    // alpha, lambda, tau, sigma, theta, cartoon
    Parameter par(20.0, 0.5, 0.25, 0.5, 1.0, atoi(argv[4]));
    // Parameter par(20.0, 0.1, 0.01, 12.5, 1.0, atoi(argv[4]));
    in.read_image(argv[1]);
    MS_Minimizer primal_dual(in, atoi(argv[3]));
    primal_dual.video(in, out, par);
    // primal_dual.real_time_minimizer(in, out, par);
    out.write_image(argv[2]);
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
    
    return 0;
}