#!/bin/sh
#
#Script for make and run program
#
#

make

# ESTIMATING BEST PARAMETERS alpha, beta, lambda:

# echo "Starting the TVL1 Test:"
# for i in "0.5" "0.6" "0.7" "0.8"
# # for i in "0.05" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5" "0.55" "0.6" "0.65" "0.7" "0.75" "0.8" "0.85" "0.9" "0.95" "1.0"
# do
# 	echo "Starting iteration "$i
# 	./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/lena_gauss_noise/tvl1/lena"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/lena_gauss_noise/tvl1/output.txt"
# 	./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/lena_sp_noise/tvl1/lena"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/lena_sp_noise/tvl1/output.txt"
# 	./iPaur -i "../../img/lena.png" -o "../../tests/lena/tvl1/lena"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/lena/tvl1/output.txt"
# 	./iPaur -i "../../img/hepburn.png" -o "../../tests/hepburn/tvl1/hepburn"$i".png" -tau 0.25 -iter 10000 -model tvl1 -lambda $i >> "../../tests/hepburn/tvl1/output.txt"
# done
# echo "TVL1 Test ended."

# echo "Starting the ROF Test:"
# for i in "0.001" "0.002" "0.003" "0.004" "0.005" "0.006" "0.007" "0.008" "0.009" "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"
# do
# 	echo "Starting iteration "$i
# 	./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/lena_gauss_noise/rof/lena"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/lena_gauss_noise/rof/output.txt"
# 	./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/lena_sp_noise/rof/lena"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/lena_sp_noise/rof/output.txt"
# 	./iPaur -i "../../img/lena.png" -o "../../tests/lena/rof/lena"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/lena/rof/output.txt"
# 	./iPaur -i "../../img/hepburn.png" -o "../../tests/hepburn/rof/hepburn"$i".png" -tau 0.25 -iter 10000 -model rof -lambda $i >> "../../tests/hepburn/rof/output.txt"
# done
# echo "ROF Test ended."

# echo "Starting the Real-Time Test:"
# for i in "2" "20" "500"
# do
# 	echo "Starting iteration "$i
# 	for j in "0.02" "0.03" "0.04" "0.07" "0.08"
# 	do
# 		./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/lena_gauss_noise/realtime/lena"$i$j".png" -tau 0.25 -iter 10000 -model realtime -alpha $i -lambda $j >> "../../tests/lena_gauss_noise/realtime/output.txt"
# 		./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/lena_sp_noise/realtime/lena"$i$j".png" -tau 0.25 -iter 10000 -model realtime -alpha $i -lambda $j >> "../../tests/lena_sp_noise/realtime/output.txt"
# 		./iPaur -i "../../img/lena.png" -o "../../tests/lena/realtime/lena"$i$j".png" -tau 0.25 -iter 10000 -model realtime -alpha $i -lambda $j >> "../../tests/lena/realtime/output.txt"
# 		./iPaur -i "../../img/hepburn.png" -o "../../tests/hepburn/realtime/hepburn"$i$j".png" -tau 0.25 -iter 10000 -model realtime -alpha $i -lambda $j >> "../../tests/hepburn/realtime/output.txt"
# 	done
# done
# echo "Real-Time Test ended."

# echo "Starting the UR Test:"
# for i in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"
# do
# 	echo "Starting iteration "$i
# 	for j in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09"
# 	do
# 		./iPaur -i "../../img/lena_gauss_noise.png" -o "../../tests/lena_gauss_noise/ur/lena"$i$j".png" -tau 0.25 -iter 10000 -model ur -alpha $i -beta $j >> "../../tests/lena_gauss_noise/ur/output.txt"
# 		./iPaur -i "../../img/lena_sp_noise.png" -o "../../tests/lena_sp_noise/ur/lena"$i$j".png" -tau 0.25 -iter 10000 -model ur -alpha $i -beta $j >> "../../tests/lena_sp_noise/ur/output.txt"
# 		./iPaur -i "../../img/lena.png" -o "../../tests/lena/ur/lena"$i$j".png" -tau 0.35 -iter 10000 -model ur -alpha $i -beta $j >> "../../tests/lena/ur/output.txt"
# 		./iPaur -i "../../img/hepburn.png" -o "../../tests/hepburn/ur/hepburn"$i$j".png" -tau 0.35 -iter 10000 -model ur -alpha $i -beta $j >> "../../tests/hepburn/ur/output.txt"
# 	done
# done
# echo "UR Test ended."

# END PARAMETER TESTING

# TESTING BEST TIME-PARAMETER FOR FASTER CONVERGENCE

for i in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.11" "0.12" "0.13" "0.14" "0.15" "0.16" "0.17" "0.18" "0.19" "0.2" "0.21" "0.22" "0.23" "0.24" "0.25" "0.26" "0.27" "0.28" "0.29" "0.3" "0.31" "0.32" "0.33" "0.34" "0.35" "0.36" "0.37" "0.38" "0.39" "0.4" "0.41" "0.42" "0.43" "0.44" "0.45" "0.46" "0.47" "0.48" "0.49" "0.5" "0.51" "0.52" "0.53" "0.54" "0.55" "0.56" "0.57" "0.58" "0.59" "0.6" "0.61" "0.62" "0.63" "0.64" "0.65" "0.66" "0.67" "0.68" "0.69" "0.7" "0.71" "0.72" "0.73" "0.74" "0.75" "0.76" "0.77" "0.78" "0.79" "0.8" "0.81" "0.82" "0.83" "0.84" "0.85" "0.86" "0.87" "0.88" "0.89" "0.9" "0.91" "0.92" "0.93" "0.94" "0.95" "0.96" "0.97" "0.98" "0.99"
do
	echo "tau = "$i
	# ./iPaur -i "../../img/lena.png" -o "../../tests/parameter_testing/rof/lena/lena"$i".png" -tau $i -iter 100000 -model rof -lambda 0.1 >> "../../tests/parameter_testing/rof/lena/output.txt"
	# ./iPaur -i "../../img/hepburn.png" -o "../../tests/parameter_testing/rof/hepburn/hepburn"$i".png" -tau $i -iter 100000 -model rof -lambda 0.07 >> "../../tests/parameter_testing/rof/hepburn/output.txt"
	./iPaur -i "../../img/van_gogh.png" -o "../../tests/parameter_testing/rof/van_gogh/van_gogh"$i".png" -tau $i -iter 100000 -model rof -lambda 0.07 >> "../../tests/parameter_testing/rof/van_gogh/output.txt"
done

# for i in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.11" "0.12" "0.13" "0.14" "0.15" "0.16" "0.17" "0.18" "0.19" "0.2" "0.21" "0.22" "0.23" "0.24" "0.25" "0.26" "0.27" "0.28" "0.29" "0.3" "0.31" "0.32" "0.33" "0.34" "0.35" "0.36" "0.37" "0.38" "0.39" "0.4" "0.41" "0.42" "0.43" "0.44" "0.45" "0.46" "0.47" "0.48" "0.49" "0.5" "0.51" "0.52" "0.53" "0.54" "0.55" "0.56" "0.57" "0.58" "0.59" "0.6" "0.61" "0.62" "0.63" "0.64" "0.65" "0.66" "0.67" "0.68" "0.69" "0.7" "0.71" "0.72" "0.73" "0.74" "0.75" "0.76" "0.77" "0.78" "0.79" "0.8" "0.81" "0.82" "0.83" "0.84" "0.85" "0.86" "0.87" "0.88" "0.89" "0.9" "0.91" "0.92" "0.93" "0.94" "0.95" "0.96" "0.97" "0.98" "0.99"
# do
# 	echo "tau = "$i
# 	./iPaur -i "../../img/lena.png" -o "../../tests/parameter_testing/ur/lena/lena"$i".png" -tau $i -iter 100000 -model ur -alpha 0.3 -beta 0.01 >> "../../tests/parameter_testing/ur/lena/output.txt"
# 	./iPaur -i "../../img/hepburn.png" -o "../../tests/parameter_testing/ur/hepburn/hepburn"$i".png" -tau $i -iter 100000 -model ur -alpha 0.3 -beta 0.01 >> "../../tests/parameter_testing/ur/hepburn/output.txt"
# 	./iPaur -i "../../img/van_gogh.png" -o "../../tests/parameter_testing/ur/van_gogh/van_gogh"$i".png" -tau $i -iter 100000 -model ur -alpha 0.3 -beta 0.01 >> "../../tests/parameter_testing/ur/van_gogh/output.txt"
# done

# for i in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.11" "0.12" "0.13" "0.14" "0.15" "0.16" "0.17" "0.18" "0.19" "0.2" "0.21" "0.22" "0.23" "0.24" "0.25" "0.26" "0.27" "0.28" "0.29" "0.3" "0.31" "0.32" "0.33" "0.34" "0.35" "0.36" "0.37" "0.38" "0.39" "0.4" "0.41" "0.42" "0.43" "0.44" "0.45" "0.46" "0.47" "0.48" "0.49" "0.5" "0.51" "0.52" "0.53" "0.54" "0.55" "0.56" "0.57" "0.58" "0.59" "0.6" "0.61" "0.62" "0.63" "0.64" "0.65" "0.66" "0.67" "0.68" "0.69" "0.7" "0.71" "0.72" "0.73" "0.74" "0.75" "0.76" "0.77" "0.78" "0.79" "0.8" "0.81" "0.82" "0.83" "0.84" "0.85" "0.86" "0.87" "0.88" "0.89" "0.9" "0.91" "0.92" "0.93" "0.94" "0.95" "0.96" "0.97" "0.98" "0.99"
# do
# 	echo "tau = "$i
# 	./iPaur -i "../../img/lena.png" -o "../../tests/parameter_testing/tvl1/lena/lena"$i".png" -tau $i -iter 100000 -model tvl1 -lambda 0.7 >> "../../tests/parameter_testing/tvl1/lena/output.txt"
# 	./iPaur -i "../../img/hepburn.png" -o "../../tests/parameter_testing/tvl1/hepburn/hepburn"$i".png" -tau $i -iter 100000 -model tvl1 -lambda 0.7 >> "../../tests/parameter_testing/tvl1/hepburn/output.txt"
# 	./iPaur -i "../../img/van_gogh.png" -o "../../tests/parameter_testing/tvl1/van_gogh/van_gogh"$i".png" -tau $i -iter 100000 -model tvl1 -lambda 0.7 >> "../../tests/parameter_testing/tvl1/van_gogh/output.txt"

# 	./iPaur -i "../../img/lena.png" -o "../../tests/parameter_testing/ur/lena/lena"$i".png" -tau $i -iter 100000 -model ur -alpha 0.3 -beta 0.01 >> "../../tests/parameter_testing/ur/lena/output.txt"
# 	./iPaur -i "../../img/hepburn.png" -o "../../tests/parameter_testing/ur/hepburn/hepburn"$i".png" -tau $i -iter 100000 -model ur -alpha 0.3 -beta 0.01 >> "../../tests/parameter_testing/ur/hepburn/output.txt"
# 	./iPaur -i "../../img/van_gogh.png" -o "../../tests/parameter_testing/ur/van_gogh/van_gogh"$i".png" -tau $i -iter 100000 -model ur -alpha 0.3 -beta 0.01 >> "../../tests/parameter_testing/ur/van_gogh/output.txt"
# done

# END TIME-PARAMETER TESTING