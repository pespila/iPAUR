#include "image.h"
#include "rgb.h"
#include "grayscale.h"
#include "parameter.h"
#include "inpainting.h"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }

    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();
    RGBImage in, out;
    in.Read(argv[1]);
    Parameter par(0.0, 64.0, 0.01, 1.0, in.GetHeight() * in.GetWidth());
    Image_Inpainting primal_dual(in, atoi(argv[3]));
    primal_dual.inpainting(in, out, par);
    out.Write(argv[2]);
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
    
    return 0;
}