#include "image.h"
#include "rgb.h"
#include "grayscale.h"
#include "parameter.h"
#include "tvl1_model.h"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }

    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();
    RGBImage in, out;
    in.read_image(argv[1]);
    Parameter par;
    TVL1_Minimizer primal_dual(in, atoi(argv[3]));
    primal_dual.tvl1_model(in, out, par);
    out.write_image(argv[2]);
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
    
    return 0;
}