#!/bin/sh
#
#Script for make and run program
#
#

make

# ESTIMATING BEST PARAMETERS lambda, nu:

# echo "Starting the TVL1 Test:"
# for i in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5"
# do
# 	echo "Starting iteration "$i
# 	./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/parameter/tvl1/lena_gauss_noise/lena_gauss_noise"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/parameter/tvl1/lena_gauss_noise/output.txt"
# 	./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/parameter/tvl1/lena_sp_noise/lena_sp_noise"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/parameter/tvl1/lena_sp_noise/output.txt"
# 	./iPaur -i "../../img/landscape.png" -o "../../tests/parameter/tvl1/landscape/landscape"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/parameter/tvl1/landscape/output.txt"
# 	./iPaur -i "../../img/van_gogh.png" -o "../../tests/parameter/tvl1/van_gogh/van_gogh"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/parameter/tvl1/van_gogh/output.txt"
# 	./iPaur -i "../../img/hepburn.png" -o "../../tests/parameter/tvl1/hepburn/hepburn"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/parameter/tvl1/hepburn/output.txt"
# 	./iPaur -i "../../img/lena.png" -o "../../tests/parameter/tvl1/lena/lena"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/parameter/tvl1/lena/output.txt"
# done
# echo "TVL1 Test ended."
# echo ""

# echo "Starting the ROF Test:"
# for i in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.2" "0.3" "0.4" "0.5"
# do
# 	echo "Starting iteration "$i
# 	./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/parameter/rof/lena_gauss_noise/lena_gauss_noise"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/parameter/rof/lena_gauss_noise/output.txt"
# 	./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/parameter/rof/lena_sp_noise/lena_sp_noise"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/parameter/rof/lena_sp_noise/output.txt"
# 	./iPaur -i "../../img/landscape.png" -o "../../tests/parameter/rof/landscape/landscape"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/parameter/rof/landscape/output.txt"
# 	./iPaur -i "../../img/van_gogh.png" -o "../../tests/parameter/rof/van_gogh/van_gogh"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/parameter/rof/van_gogh/output.txt"
# 	./iPaur -i "../../img/hepburn.png" -o "../../tests/parameter/rof/hepburn/hepburn"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/parameter/rof/hepburn/output.txt"
# 	./iPaur -i "../../img/lena.png" -o "../../tests/parameter/rof/lena/lena"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/parameter/rof/lena/output.txt"
# done
# echo "ROF Test ended."
# echo ""

# echo "Starting the Real-Time Test:"
# for i in "2" "20" "500"
# do
# 	echo "Starting iteration "$i
# 	for j in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
# 	do
# 		echo "Starting iteration "$j
# 		./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/parameter/realtime/lena_gauss_noise/"$j"lena_gauss_noise"$i".png" -iter 10000 -model realtime -lambda $i -nu $j # >> "../../tests/parameter/realtime/lena_gauss_noise/output.txt"
# 		./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/parameter/realtime/lena_sp_noise/"$j"lena_sp_noise"$i".png" -iter 10000 -model realtime -lambda $i -nu $j # >> "../../tests/parameter/realtime/lena_sp_noise/output.txt"
# 		./iPaur -i "../../img/landscape.png" -o "../../tests/parameter/realtime/landscape/"$j"landscape"$i".png" -iter 10000 -model realtime -lambda $i -nu $j # >> "../../tests/parameter/realtime/landscape/output.txt"
# 		./iPaur -i "../../img/van_gogh.png" -o "../../tests/parameter/realtime/van_gogh/"$j"van_gogh"$i".png" -iter 10000 -model realtime -lambda $i -nu $j # >> "../../tests/parameter/realtime/van_gogh/output.txt"
# 		./iPaur -i "../../img/hepburn.png" -o "../../tests/parameter/realtime/hepburn/"$j"hepburn"$i".png" -iter 10000 -model realtime -lambda $i -nu $j # >> "../../tests/parameter/realtime/hepburn/output.txt"
# 		./iPaur -i "../../img/lena.png" -o "../../tests/parameter/realtime/lena/"$j"lena"$i".png" -iter 10000 -model realtime -lambda $i -nu $j # >> "../../tests/parameter/realtime/lena/output.txt"
# 	done
# done
# echo "Real-Time Test ended."
# echo ""

# echo "Starting the Segmentation Test:"
# echo "Starting iteration "$i
# for j in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
# do
# 	echo "Starting iteration "$j
	# ./iPaur -i "../../img/blue.png" -o "../../tests/parameter/segmentation/blue/"$j"blue.png" -iter 1000 -model realtime -lambda 500 -nu $j >> "../../tests/parameter/segmentation/blue/output.txt"
	# ./iPaur -i "../../img/peacock-feather.png" -o "../../tests/parameter/segmentation/peacock-feather/"$j"peacock-feather.png" -iter 1000 -model realtime -lambda 500 -nu $j >> "../../tests/parameter/segmentation/peacock-feather/output.txt"
	# ./iPaur -i "../../img/squirrel.png" -o "../../tests/parameter/segmentation/squirrel/"$j"squirrel.png" -iter 1000 -model realtime -lambda 500 -nu $j >> "../../tests/parameter/segmentation/squirrel/output.txt"
# 	./iPaur -i "../../img/keating.jpg" -o "../../tests/parameter/segmentation/keating/"$j"keating.jpg" -iter 1000 -model realtime -lambda 500 -nu $j >> "../../tests/parameter/segmentation/keating/output.txt"
# done
# echo "Segmentation Test ended."
# echo ""

# echo "Starting the INPAINT Test:"
# for i in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.2" "0.3" "0.4" "0.5"
# do
# 	echo "lambda = "$i
# 	./iPaur -i "../../img/lena_inpaint.png" -o "../../tests/parameter/inpaint/lena_inpaint/psnr/lena_inpaint"$i".png" -iter 10000 -model inpaint -lambda $i >> "../../tests/parameter/inpaint/lena_inpaint/output.txt"
# 	./iPaur -o "../../test.png" -i "../../img/lena.png" -c "../../tests/parameter/inpaint/lena_inpaint/psnr/lena_inpaint"$i".png" -model psnr >> "../../tests/parameter/inpaint/lena_inpaint/psnr/psnr.txt"
# done
# echo "INPAINT Test ended."
# echo ""

# END PARAMETER TESTING

# ESTIMATING PSNR for Denoising and Inpainting

# echo "Starting the TVL1 Test:"
# for i in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5"
# do
# 	echo "Starting iteration "$i
# 	./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/parameter/tvl1/lena_gauss_noise/psnr/lena_gauss_noise"$i".png" -tau 0.97 -iter 10000 -model tvl1 -lambda $i
# 	./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/parameter/tvl1/lena_sp_noise/psnr/lena_sp_noise"$i".png" -tau 0.97 -iter 10000 -model tvl1 -lambda $i
# 	./iPaur -o "../../test.png" -i "../../img/lena.png" -c "../../tests/parameter/tvl1/lena_gauss_noise/psnr/lena_gauss_noise"$i".png" -model psnr -lambda $i >> "../../tests/parameter/tvl1/lena_gauss_noise/psnr/psnr.txt"
# 	./iPaur -o "../../test.png" -i "../../img/lena.png" -c "../../tests/parameter/tvl1/lena_sp_noise/psnr/lena_sp_noise"$i".png" -model psnr -lambda $i >> "../../tests/parameter/tvl1/lena_sp_noise/psnr/psnr.txt"
# done
# echo "TVL1 Test ended."
# echo ""

# echo "Starting the ROF Test:"
# for i in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.2" "0.3" "0.4" "0.5"
# do
# 	echo "Starting iteration "$i
# 	./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/parameter/rof/lena_gauss_noise/psnr/lena_gauss_noise"$i".png" -tau 0.73 -iter 10000 -model rof -lambda $i
# 	./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/parameter/rof/lena_sp_noise/psnr/lena_sp_noise"$i".png" -tau 0.73 -iter 10000 -model rof -lambda $i
# 	./iPaur -o "../../test.png" -i "../../img/lena.png" -c "../../tests/parameter/rof/lena_gauss_noise/psnr/lena_gauss_noise"$i".png" -model psnr -lambda $i >> "../../tests/parameter/rof/lena_gauss_noise/psnr/psnr.txt"
# 	./iPaur -o "../../test.png" -i "../../img/lena.png" -c "../../tests/parameter/rof/lena_sp_noise/psnr/lena_sp_noise"$i".png" -model psnr -lambda $i >> "../../tests/parameter/rof/lena_sp_noise/psnr/psnr.txt"
# done
# echo "ROF Test ended."
# echo ""

# echo "Starting the Real-Time Test:"
# for i in "2" "20" "500"
# do
# 	echo "Starting iteration "$i
# 	for j in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
# 	do
# 		echo "Starting iteration "$j
# 		./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/parameter/realtime/lena_gauss_noise/psnr/"$j"lena_gauss_noise"$i".png" -iter 10000 -model realtime -lambda $i -nu $j
# 		./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/parameter/realtime/lena_sp_noise/psnr/"$j"lena_sp_noise"$i".png" -iter 10000 -model realtime -lambda $i -nu $j
# 		./iPaur -o "../../test.png" -i "../../img/lena.png" -c "../../tests/parameter/realtime/lena_gauss_noise/psnr/"$j"lena_gauss_noise"$i".png" -model psnr -lambda $i >> "../../tests/parameter/realtime/lena_gauss_noise/psnr/psnr.txt"
# 		./iPaur -o "../../test.png" -i "../../img/lena.png" -c "../../tests/parameter/realtime/lena_sp_noise/psnr/"$j"lena_sp_noise"$i".png" -model psnr -lambda $i >> "../../tests/parameter/realtime/lena_sp_noise/psnr/psnr.txt"
# 	done
# done
# echo "Real-Time Test ended."
# echo ""

# END Denoising and Inpainting testing

# TESTING BEST TIME-PARAMETER FOR FASTER CONVERGENCE

# echo "Starting the TVL1 Test:"
# for i in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.11" "0.12" "0.13" "0.14" "0.15" "0.16" "0.17" "0.18" "0.19" "0.2" "0.21" "0.22" "0.23" "0.24" "0.25" "0.26" "0.27" "0.28" "0.29" "0.3" "0.31" "0.32" "0.33" "0.34" "0.35" "0.36" "0.37" "0.38" "0.39" "0.4" "0.41" "0.42" "0.43" "0.44" "0.45" "0.46" "0.47" "0.48" "0.49" "0.5" "0.51" "0.52" "0.53" "0.54" "0.55" "0.56" "0.57" "0.58" "0.59" "0.6" "0.61" "0.62" "0.63" "0.64" "0.65" "0.66" "0.67" "0.68" "0.69" "0.7" "0.71" "0.72" "0.73" "0.74" "0.75" "0.76" "0.77" "0.78" "0.79" "0.8" "0.81" "0.82" "0.83" "0.84" "0.85" "0.86" "0.87" "0.88" "0.89" "0.9" "0.91" "0.92" "0.93" "0.94" "0.95" "0.96" "0.97" "0.98" "0.99"
# do
	# echo "tau = "$i
	# ./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/parameter/tvl1/lena_gauss_noise/lena_gauss_noise"$i".png" -tau $i -iter 10000 -model tvl1 -lambda 0.7 >> "../../tests/parameter/tvl1/lena_gauss_noise/tau.txt"
	# ./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/parameter/tvl1/lena_sp_noise/lena_sp_noise"$i".png" -tau $i -iter 10000 -model tvl1 -lambda 0.7 >> "../../tests/parameter/tvl1/lena_sp_noise/tau.txt"
	# ./iPaur -i "../../img/landscape.png" -o "../../tests/parameter/tvl1/landscape/landscape"$i".png" -tau $i -iter 10000 -model tvl1 -lambda 1.2 >> "../../tests/parameter/tvl1/landscape/tau.txt"
	# ./iPaur -i "../../img/van_gogh.png" -o "../../tests/parameter/tvl1/van_gogh/van_gogh"$i".png" -tau $i -iter 10000 -model tvl1 -lambda 1.2 >> "../../tests/parameter/tvl1/van_gogh/tau.txt"
	# ./iPaur -i "../../img/hepburn.png" -o "../../tests/parameter/tvl1/hepburn/hepburn"$i".png" -tau $i -iter 10000 -model tvl1 -lambda 1.2 >> "../../tests/parameter/tvl1/hepburn/tau.txt"
	# ./iPaur -i "../../img/lena.png" -o "../../tests/parameter/tvl1/lena/lena"$i".png" -tau $i -iter 10000 -model tvl1 -lambda 0.7 >> "../../tests/parameter/tvl1/lena/tau.txt"
# done
# echo "TVL1 Test ended."
# echo ""

# echo "Starting the ROF Test:"
# for i in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.11" "0.12" "0.13" "0.14" "0.15" "0.16" "0.17" "0.18" "0.19" "0.2" "0.21" "0.22" "0.23" "0.24" "0.25" "0.26" "0.27" "0.28" "0.29" "0.3" "0.31" "0.32" "0.33" "0.34" "0.35" "0.36" "0.37" "0.38" "0.39" "0.4" "0.41" "0.42" "0.43" "0.44" "0.45" "0.46" "0.47" "0.48" "0.49" "0.5" "0.51" "0.52" "0.53" "0.54" "0.55" "0.56" "0.57" "0.58" "0.59" "0.6" "0.61" "0.62" "0.63" "0.64" "0.65" "0.66" "0.67" "0.68" "0.69" "0.7" "0.71" "0.72" "0.73" "0.74" "0.75" "0.76" "0.77" "0.78" "0.79" "0.8" "0.81" "0.82" "0.83" "0.84" "0.85" "0.86" "0.87" "0.88" "0.89" "0.9" "0.91" "0.92" "0.93" "0.94" "0.95" "0.96" "0.97" "0.98" "0.99"
# do
# 	echo "tau = "$i
# 	./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/parameter/rof/lena_gauss_noise/lena_gauss_noise"$i".png" -tau $i -iter 10000 -model rof -lambda 0.03 >> "../../tests/parameter/rof/lena_gauss_noise/tau.txt"
	# ./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/parameter/rof/lena_sp_noise/lena_sp_noise"$i".png" -tau $i -iter 10000 -model rof -lambda 0.01 >> "../../tests/parameter/rof/lena_sp_noise/tau.txt"
	# ./iPaur -i "../../img/landscape.png" -o "../../tests/parameter/rof/landscape/landscape"$i".png" -tau $i -iter 10000 -model rof -lambda 0.1 >> "../../tests/parameter/rof/landscape/tau.txt"
	# ./iPaur -i "../../img/van_gogh.png" -o "../../tests/parameter/rof/van_gogh/van_gogh"$i".png" -tau $i -iter 10000 -model rof -lambda 0.1 >> "../../tests/parameter/rof/van_gogh/tau.txt"
	# ./iPaur -i "../../img/hepburn.png" -o "../../tests/parameter/rof/hepburn/hepburn"$i".png" -tau $i -iter 10000 -model rof -lambda 0.1 >> "../../tests/parameter/rof/hepburn/tau.txt"
	# ./iPaur -i "../../img/lena.png" -o "../../tests/parameter/rof/lena/lena"$i".png" -tau $i -iter 10000 -model rof -lambda 0.1 >> "../../tests/parameter/rof/lena/tau.txt"
# done
# echo "ROF Test ended."
# echo ""

# echo "Starting the INPAINT Test:"
# for i in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.11" "0.12" "0.13" "0.14" "0.15" "0.16" "0.17" "0.18" "0.19" "0.2" "0.21" "0.22" "0.23" "0.24" "0.25" "0.26" "0.27" "0.28" "0.29" "0.3" "0.31" "0.32" "0.33" "0.34" "0.35" "0.36" "0.37" "0.38" "0.39" "0.4" "0.41" "0.42" "0.43" "0.44" "0.45" "0.46" "0.47" "0.48" "0.49" "0.5" "0.51" "0.52" "0.53" "0.54" "0.55" "0.56" "0.57" "0.58" "0.59" "0.6" "0.61" "0.62" "0.63" "0.64" "0.65" "0.66" "0.67" "0.68" "0.69" "0.7" "0.71" "0.72" "0.73" "0.74" "0.75" "0.76" "0.77" "0.78" "0.79" "0.8" "0.81" "0.82" "0.83" "0.84" "0.85" "0.86" "0.87" "0.88" "0.89" "0.9" "0.91" "0.92" "0.93" "0.94" "0.95" "0.96" "0.97" "0.98" "0.99"
# do
# 	echo "tau = "$i
# 	./iPaur -i "../../img/lena_inpaint.png" -o "../../tests/parameter/inpaint/lena_inpaint/lena_inpaint"$i".png" -iter 10000 -model inpaint -lambda 0.3 -tau $i >> "../../tests/parameter/inpaint/lena_inpaint/tau.txt"
# done
# echo "INPAINT Test ended."
# echo ""