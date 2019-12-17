#! /bin/bash

readonly OLD="result/summed_float_"
readonly NEW="result/summed_hist_float_"

alpha=(0.0 0.20 0.40 0.60 0.80 1.0)

for a in ${alpha[@]}
do 
    for x in $(seq 5 9)
    do
	old="${OLD}${a}_${x}.txt"
	new="${NEW}${a}_${x}.txt"

	echo $old
	echo $new
	mv $old $new

    done
done
