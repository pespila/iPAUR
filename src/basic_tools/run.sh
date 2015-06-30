#!/bin/sh
#
#Script for make and run program
#
#

make

./ImageProcessing "../img/lena_noise.png" "../img/lena_new.png"

# _________________________________________________________________________________________________________
# TI pictures
# _________________________________________________________________________________________________________
# for j in "gauss" "mean" "canny" "laplace" "erosion" "Ierosion" "dilatation" "Idilatation" "close" "Iclose" "inverse"
# do
	# for (( i = 1; i < 10; i++ ));
	# do
	# 	./ImageProcessing "../img/wafer/$i.JPG" "../img/waferTest/$i-test3.JPG" # "$j" "3"
	# 	echo "$i"
	# done
# done
# _________________________________________________________________________________________________________


# _________________________________________________________________________________________________________
# Linear Filter
# _________________________________________________________________________________________________________
# for i in "gauss" "duto" "mean"
# do
# 	./ImageProcessing "../img/lena_noisy.png" "../img/lena_noisy_$i.png" "$i" "3"
# done
# _________________________________________________________________________________________________________


# _________________________________________________________________________________________________________
# Edge Detection
# _________________________________________________________________________________________________________
# for i in "log" "canny" "log" "sobel" "prewitt" "roberts"
# do
	# ./ImageProcessing "../img/ball.png" "../img/ball_$i.png" "$i" "3"
	# ./ImageProcessing "../img/fcb.png" "../img/fcb_$i.png" "$i" "15"
	# ./ImageProcessing "../img/apple.png" "../img/apple_$i.png" "$i" "3"
	# ./ImageProcessing "../img/image.png" "../img/image_$i.png" "$i" "3"
	# ./ImageProcessing "../img/lena_noisy.png" "../img/lena_$i.png" "$i" "9"
# done
# _________________________________________________________________________________________________________

# for i in "transform_rgb" "transform_hsi" "transform_ycrcb";
# do
# 	./ImageProcessing "../img/$i.png" "../img/$i_new.png"
# done
# ./ImageProcessing "../img/miche.jpg" "../img/miche_new.jpg" "../img/miche_original.jpg"
# ./ImageProcessing "../img/lena_color_new.png" "../img/lena_color_edge.png" "../img/lena_color_original.png"
# ./ImageProcessing "../img/miche.jpg" "../img/miche_new.jpg"
# ./ImageProcessing "../img/wafer_original.png" "../img/wafer_original_new.png"
# ./ImageProcessing "../img/wafer.png" "../img/wafer_new.png"

# ./ImageProcessing "../img/la_dama.jpg" "../img/belladonna.jpg"
# ./ImageProcessing "../img/trash.png" "../img/trash_new.png"
# ./ImageProcessing "../img/pyramide_new.png" "../img/pyramide_new2.png"
# ./ImageProcessing "../img/pyramide_new1.png" "../img/pyramide_new2.png"
# ./ImageProcessing "../img/pyramide.png" "../img/pyramide_new0.png"
# ./ImageProcessing "../img/lena.png" "../img/lena_new.png" "gauss" "3"
# ./ImageProcessing "../img/belladonna.jpg" "../img/belladonna_new.jpg"

# ./ImageProcessing "../img/M.png" "../img/M_piv.png"
# ./ImageProcessing "../img/lena_color.png" "../img/lena_face.png"
# ./ImageProcessing "../img/jobs.jpg" "../img/jobs_face.png"
# ./ImageProcessing "../img/kd.jpg" "../img/kd_face.png"
# ./ImageProcessing "../img/miche.jpg" "../img/miche_face.png"

# _________________________________________________________________________________________________________
# Morphological Filter
# _________________________________________________________________________________________________________
# for i in "inverse" "erosion" "dilatation" "open" "close" "whitetophat" "blacktophat" #"hitormiss"
# do
# 	./ImageProcessing "../img/lena.png" "../img/lena_$i.png" "$i" "3"
# 	./ImageProcessing "../img/one_standard.png" "../img/one_standard_$i.png" "$i" "3"
# done
# _________________________________________________________________________________________________________


# _________________________________________________________________________________________________________
# Testing with inversion, erosion, reinversion
# _________________________________________________________________________________________________________
# ./ImageProcessing "../img/image.png" "../img/image_inverse.png" "inverse" "3"
# ./ImageProcessing "../img/image_inverse.png" "../img/image_inverse_erosion.png" "erosion" "3"
# ./ImageProcessing "../img/image_inverse_erosion.png" "../img/image_reinverse_erosion.png" "inverse" "3"
# ./ImageProcessing "../img/lena.png" "../img/lena_inverse.png" "inverse" "3"
# ./ImageProcessing "../img/lena_inverse.png" "../img/lena_inverse_erosion.png" "erosion" "3"
# ./ImageProcessing "../img/lena_inverse_erosion.png" "../img/lena_reinverse_erosion.png" "inverse" "3"
# _________________________________________________________________________________________________________


# _________________________________________________________________________________________________________
# Valgrind
# _________________________________________________________________________________________________________
# ./main "../img/lena.png" "../img/lena_systemTest.png" "canny"
# valgrind ./ImageProcessing "../img/test.png" "../img/test_test.png" "gauss" "3" > testDrive.txt 2>&1
# valgrind --leak-check=full ./ImageProcessing "../img/lena_noisy.png" "../img/lena_systemTest.png" "laplace" "3" > testDrive.txt 2>&1
# valgrind --track-origins=yes ./ImageProcessing "../img/test.png" "../img/test_systemTest.png" "canny" "3" > testDrive.txt 2>&1
# _________________________________________________________________________________________________________

# rm *.o