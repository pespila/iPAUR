#!/bin/sh
#
#Script for make and run program
#
#

# FIT THE (GOOD DEFAULT) PARAMETER TO THE ALGORTIHMS
#	- Huber-ROF-Model: alpha = 0.05, lambda = 32.0, tau = 0.01, sigma = 0.0, theta = 1.0, cartoon = -1
#	- Image Inpainting: alpha = 0.05, lambda = 32.0, tau = 0.01, sigma = 0.0, theta = 1.0, cartoon = -1
#	- TVL1-Model: alpha = 0.05, lambda = 0.7, tau = 0.35, sigma = 1.0 / (0.35 * 8.0), theta = 1.0, cartoon = -1
#	- Real-Time-Minimizer: alpha = 20.0, lambda = 0.1, tau = 0.25, sigma = 0.5, theta = 1.0, cartoon = 1/0
#

make

echo "Starting the TVL1 Test:"
for i in "0.5" "0.6" "0.7" "0.8"
# for i in "0.05" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5" "0.55" "0.6" "0.65" "0.7" "0.75" "0.8" "0.85" "0.9" "0.95" "1.0"
do
	echo "Starting iteration "$i
	./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/lena_gauss_noise/tvl1/lena"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/lena_gauss_noise/tvl1/output.txt"
	./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/lena_sp_noise/tvl1/lena"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/lena_sp_noise/tvl1/output.txt"
	./iPaur -i "../../img/lena.png" -o "../../tests/lena/tvl1/lena"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/lena/tvl1/output.txt"
	./iPaur -i "../../img/hepburn.png" -o "../../tests/hepburn/tvl1/hepburn"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/hepburn/tvl1/output.txt"
done
echo "TVL1 Test ended."

echo "Starting the ROF Test:"
for i in "0.001" "0.002" "0.003" "0.004" "0.005" "0.006" "0.007" "0.008" "0.009" "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
do
	echo "Starting iteration "$i
	./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/lena_gauss_noise/rof/lena"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/lena_gauss_noise/rof/output.txt"
	./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/lena_sp_noise/rof/lena"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/lena_sp_noise/rof/output.txt"
	./iPaur -i "../../img/lena.png" -o "../../tests/lena/rof/lena"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/lena/rof/output.txt"
	./iPaur -i "../../img/hepburn.png" -o "../../tests/hepburn/rof/hepburn"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/hepburn/rof/output.txt"
done
echo "ROF Test ended."

echo "Starting the Real-Time Test:"
for i in "2" "20" "500"
do
	echo "Starting iteration "$i
	for j in "0.02" "0.03" "0.04" "0.07" "0.08"
	do
		./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/lena_gauss_noise/realtime/lena"$i$j".png" -tau 0.25 -iter 10000 -model realtime -alpha $i -lambda $j >> "../../tests/lena_gauss_noise/realtime/output.txt"
		./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/lena_sp_noise/realtime/lena"$i$j".png" -tau 0.25 -iter 10000 -model realtime -alpha $i -lambda $j >> "../../tests/lena_sp_noise/realtime/output.txt"
		./iPaur -i "../../img/lena.png" -o "../../tests/lena/realtime/lena"$i$j".png" -tau 0.25 -iter 10000 -model realtime -alpha $i -lambda $j >> "../../tests/lena/realtime/output.txt"
		./iPaur -i "../../img/hepburn.png" -o "../../tests/hepburn/realtime/hepburn"$i$j".png" -tau 0.25 -iter 10000 -model realtime -alpha $i -lambda $j >> "../../tests/hepburn/realtime/output.txt"
	done
done
echo "Real-Time Test ended."

echo "Starting the UR Test:"
for i in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"
do
	echo "Starting iteration "$i
	for j in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09"
	do
		./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/lena_gauss_noise/ur/lena"$i$j".png" -tau 0.25 -iter 10000 -model ur -alpha $i -beta $j >> "../../tests/lena_gauss_noise/ur/output.txt"
		./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/lena_sp_noise/ur/lena"$i$j".png" -tau 0.25 -iter 10000 -model ur -alpha $i -beta $j >> "../../tests/lena_sp_noise/ur/output.txt"
		./iPaur -i "../../img/lena.png" -o "../../tests/lena/ur/lena"$i$j".png" -tau 0.35 -iter 10000 -model ur -alpha $i -beta $j >> "../../tests/lena/ur/output.txt"
		./iPaur -i "../../img/hepburn.png" -o "../../tests/hepburn/ur/hepburn"$i$j".png" -tau 0.35 -iter 10000 -model ur -alpha $i -beta $j >> "../../tests/hepburn/ur/output.txt"
	done
done
echo "UR Test ended."