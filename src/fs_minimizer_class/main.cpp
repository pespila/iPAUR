#include "image.h"
#include "rgb.h"
#include "grayscale.h"
#include "parameter.h"
#include "fast_minimizer.h"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }

    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();
    RGBImage in, out;
    Parameter par;
    in.read_image(argv[1]);
    MS_Minimizer primal_dual(in, atoi(argv[3]));
    primal_dual.fast_minimizer(in, out, par);
    out.write_image(argv[2]);
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
    
    return 0;
}