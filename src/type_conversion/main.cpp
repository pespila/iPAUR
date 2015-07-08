#include "image.h"
#include "rgb.h"
#include "grayscale.h"
#include "typeconversion.h"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }

    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();
    RGBImage in;
    HSIImage out;
    in.read_image(argv[1]);
    TypeConversion typ(in);
    typ.rgb2hsi(in, out);
    out.write_image(argv[2]);
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
    
    return 0;
}