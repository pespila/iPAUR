#!/bin/sh
#
#Script for make and run program
#
#

make

# ./ImageProcessing "../img/pyramide.png" "../img/pyramide_new.png"
./ImageProcessing "../img/lena_noisy.png" "../img/lena_noisy_new.png"
# ./ImageProcessing "../img/lena_noise.png" "../img/lena_noise_new.png"
# ./ImageProcessing "../img/lena.png" "../img/lena_new.png"
# ./ImageProcessing "../img/belladonna.jpg" "../img/belladonna_new.jpg"
# ./ImageProcessing "../img/image_noisy.png" "../img/image_noisy_new.png"
# ./ImageProcessing "../img/wafer.png" "../img/wafer_new.png"
# ./ImageProcessing "../img/bremen.jpg" "../img/bremen_new.jpg"
# for i in "transform_rgb" "transform_hsi" "transform_ycrcb";
# do
# 	./ImageProcessing "../img/$i.png" "../img/$i_new.png"
# done

# rm *.o