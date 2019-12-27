#!bin/bash

#Input
readonly DATA="$HOME/Desktop/data/kits19"
readonly SAVE="$HOME/Desktop/data/box"

numArr=(000 001 003 004 006 007 009 010 014 015 017 018 019 020 022 023 027 031 032 033 037 039 040 043 049 050 052 054 062 063 064 065 071 072 075 076 077 081 082 083 085 091 093 094 096 097 100 101 103 106 115 120 121 123 124 125 127 128 129 132 136 137 138 140 141 146 150 152 153 155 156 158 164 167 173 174 175 182 188 190 191 193 198 201 203 205 )

for number in ${numArr[@]}
do
	data="${DATA}/case_00${number}"
	save="${SAVE}/case_00${number}"

	echo $data
	echo $save

	
	python3 clipKidney.py ${data} ${save} 

	if [ $? -eq 0 ]; then
		echo "case_00${number} done."
	
	else
		echo "case_00${number}" >> fail/clipKidney.txt
		echo "case_00${number} failed"
	
	fi
	    

done
