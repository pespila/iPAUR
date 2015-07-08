#include "image.h"
#include "rgb.h"
#include "rgba.h"
#include "hsi.h"
#include "ycrcb.h"
#include "grayscale.h"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }

    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();
    RGBImage in;
    in.Read(argv[1]);    
    in.Write(argv[2]);
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
    
    return 0;
}