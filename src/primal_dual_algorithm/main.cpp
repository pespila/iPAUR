#include "image.h"
#include "rgb.h"
#include "grayscale.h"
#include "parameter.h"
#include "primal_dual_algorithm.h"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }

    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();
    GrayscaleImage in, out;
    Parameter par;
    in.Read(argv[1]);
    Primal_Dual primal_dual(in, atoi(argv[3]), atoi(argv[4]));
    primal_dual.primal_dual_algorithm(in, out, par);
    out.Write(argv[2]);
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
    
    return 0;
}