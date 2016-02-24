#!/bin/sh
#
#Script for make and run program
#
#

make

# -level 16 -repeats 1000 -nu 0.01 -lambda 0.11 > "../results/lena.txt"
# -level 16 -repeats 1000 -nu 0.01 -lambda 0.11 > "../results/lena_noise.txt"
# -level 20 -repeats 100000 -nu 0.01 -lambda 0.11 > "../results/lena_noisy.txt"
# -level 20 -repeats 1 -nu 0.01 -lambda 0.1 > "../results/hepburn.txt"
# -level 8 -repeats 10000 -nu 0.01 -lambda 0.1 > "../results/ladama/data.txt"
# -level 8 -repeats 10000 -nu 0.001 -lambda 0.1 > "../results/marylin/data.txt"
# -level 8 -repeats 10000 -nu 0.01 -lambda 0.11 > "../results/synth_gauss/data.txt"
# -level 32 -repeats 10000 -nu 0.01 -lambda 0.1 > "../results/crack_tip/data.txt"
# -level 16 -repeats 10000 -nu 0.0001 -lambda 0.1 > "../results/synth/data.txt"

# file="synth_gauss"
# ./primaldual -i $img$file".png" -o $res$file"/"$file".png" -data $nrj -parm $par -level 16 -repeats 10000 -nu 0.01 -lambda 0.11

# nrj="data.txt"
img="../../img/"
res="./results/"
file="synth"

for i in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1"
do
	for j in "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1"
	do
		echo $i" "$j
		par=$res$file"/parameter.txt"
		./primaldual -i $img$file".png" -o $res$file"/"$i$j$file".png" -parm $par -level 32 -repeats 1000 -nu $i -lambda $j -gray
	done
done

# for file in "synth" "lena" "lena_noisy" "hepburn" "ladama" "marylin" "synth_gauss" "crack_tip" "inpaint";
# for file in "hepburnc" "blue" "gaudi" "lake" "landscape" "marble" "squirrel" "van_gogh";
# for file in "synth"
# do
# 	for i in `seq 1 32`
# 	do
# 		echo $i
# 		par=$res$file"/parameter.txt"
# 		# out=$res$file"/dual_energy.png"
# 		# nrj=$res$file"/data.txt"
# 		./primaldual -i $img$file".png" -o $res$file"/"$i$file".png" -parm $par -level $i -repeats 1000 -nu 0.01 -lambda 0.1 -gray
# 		# gnuplot -e "outfile='"$out"'" -e "datafile='"$nrj"'" plot.gpl
# 	done
# done
# rm data.txt
# rm ./primaldual