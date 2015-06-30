#include "image.h"
#include "rgb.h"
#include "grayscale.h"
#include "fast_minimizer.h"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }

    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();
    RGBImage Mountain, MountainNew;
    // GrayscaleImage Mountain, MountainNew;
    Mountain.read_image(argv[1]);
    fast_minimizer(Mountain, MountainNew, atoi(argv[3]), atoi(argv[4]));
    MountainNew.write_image(argv[2]);
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
    
    return 0;
}