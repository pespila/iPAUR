#include "image.h"
#include "rgb.h"
#include "grayscale.h"
#include "morphologicalfilter.h"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }

    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();
    GrayscaleImage in, out;
    in.read_image(argv[1]);
    MorphologicalFilter filter(in);
    // filter.Erosion(in, in, 1);
    // filter.Dilatation(in, out, 1);
    filter.Median(in, out, 1);
    out.write_image(argv[2]);
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
    
    return 0;
}