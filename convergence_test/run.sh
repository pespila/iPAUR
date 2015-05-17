#!/bin/sh
#
#Script for make and run program
#
#

make

# ./ImageProcessing "../img/pyramide.png" "../img/pyramide_new.png" "2000" "../plot/data_pyramide.dat"
# ./ImageProcessing "../img/lena_noisy.png" "../img/lena_noisy_new.png" "1000" "../plot/data_lena_noisy.dat"
# ./ImageProcessing "../img/lena_noise.png" "../img/lena_noise_new.png" "1000" "../plot/data_lena_noise.dat"
# ./ImageProcessing "../img/image_noisy.png" "../img/image_noisy_new.png" "1000" "../plot/data_image_noisy.dat"
./ImageProcessing "../img/lena.png" "../img/lena_new.png" "2000" "../plot/data_lena.dat"
# ./ImageProcessing "../img/belladonna.jpg" "../img/belladonna_new.jpg" "1000" "../plot/data_belladonna.dat"
# ./ImageProcessing "../img/image.png" "../img/image_new.png" "1000" "../plot/data_image.dat"

rm *.o