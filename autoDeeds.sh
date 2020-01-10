#!bin/bash

#Input
readonly DATA="$HOME/Desktop/data/box/AABB/black"
readonly SAVE="$HOME/Desktop/data/box"

numArr=(000 001 003 004 006 007 009 010 014 015 017 018 019 020 022 023 027 031 032 033 037 039 040 043 049 050 052 054 062 063 064 065 071 072 075 076 077 081 082 083 085 091 093 094 096 097 100 101 103 106 115 120 121 123 124 125 127 128 129 132 136 137 138 140 141 146 150 152 153 155 156 158 164 167 173 174 175 182 188 190 191 193 198 201 203 205 )

date >> fail/deeds.txt

for number in ${numArr[@]}
do
	leftData="${DATA}/case_00${number}/image_left.nii.gz"
	rightData="${DATA}/case_00${number}/image_right.nii.gz"

	leftNonLinearSave="${SAVE}/nonLinear/case_00${number}/image_left.nii.gz"
	rightNonLinearSave="${SAVE}/nonLinear/case_00${number}/image_right.nii.gz"
	leftWithLinearSave="${SAVE}/withLinear/case_00${number}/image_left.nii.gz"
	rightWithLinearSave="${SAVE}/withLinear/case_00${number}/image_right.nii.gz"

	mkdir -p "${SAVE}/nonLinear/case_00${number}"
	mkdir -p "${SAVE}/withLinear/case_00${number}"


	echo $leftData
	echo $rightData
	echo $leftNonLinearSave
	echo $rightNonLinearSave
	echo $leftWithLinearSave
	echo $rightWithLinearSave

	../deedsBCV/linearBCV -F $leftData -M $rightData -O right
	../deedsBCV/linearBCV -F $rightData -M $leftData -O left

	../deedsBCV/deedsBCV -F $leftData -M $rightData -O right_nonLinear
	../deedsBCV/deedsBCV -F $leftData -M $rightData -A right_matrix.txt -O right_withLinear

	../deedsBCV/deedsBCV -F $rightData -M $leftData -O left_nonLinear
	../deedsBCV/deedsBCV -F $rightData -M $leftData -A left_matrix.txt -O left_withLinear

	mv right_nonLinear_deformed.nii.gz $rightNonLinearSave
	mv right_withLinear_deformed.nii.gz $rightWithLinearSave
	mv left_nonLinear_deformed.nii.gz $leftNonLinearSave
	mv left_withLinear_deformed.nii.gz $leftWithLinearSave

	if [ $? -eq 0 ]; then
		echo "case_00${number} done."
	
	else
		echo "case_00${number}" >> fail/deeds.txt
		echo "case_00${number} failed"
	
	fi
	    

done

rm right*
rm left*
