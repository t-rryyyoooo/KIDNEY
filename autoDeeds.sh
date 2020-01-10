#!bin/bash

#Input
readonly DATA="$HOME/Desktop/data/box/AABB/black"
readonly SAVE="$HOME/Desktop/data/box"

numArr=(000 001 003 004 006 007 009 010 014 015 017 018 019 020 022 023 027 031 032 033 037 039 040 043 049 050 052 054 062 063 064 065 071 072 075 076 077 081 082 083 085 091 093 094 096 097 100 101 103 106 115 120 121 123 124 125 127 128 129 132 136 137 138 140 141 146 150 152 153 155 156 158 164 167 173 174 175 182 188 190 191 193 198 201 203 205 )

date >> fail/deeds.txt

for number in ${numArr[@]}
do
	leftImageData="${DATA}/case_00${number}/image_left.nii.gz"
	rightImageData="${DATA}/case_00${number}/image_right.nii.gz"
	leftLabelData="${DATA}/case_00${number}/label_left.nii.gz"
	rightLabelData="${DATA}/case_00${number}/label_right.nii.gz"


	leftImageNonLinearSave="${SAVE}/nonLinear/case_00${number}/image_left.nii.gz"
	rightImageNonLinearSave="${SAVE}/nonLinear/case_00${number}/image_right.nii.gz"
	leftImageWithLinearSave="${SAVE}/withLinear/case_00${number}/image_left.nii.gz"
	rightImageWithLinearSave="${SAVE}/withLinear/case_00${number}/image_right.nii.gz"
	leftLabelNonLinearSave="${SAVE}/nonLinear/case_00${number}/label_left.nii.gz"
	rightLabelNonLinearSave="${SAVE}/nonLinear/case_00${number}/label_right.nii.gz"
	leftLabelWithLinearSave="${SAVE}/withLinear/case_00${number}/label_left.nii.gz"
	rightLabelWithLinearSave="${SAVE}/withLinear/case_00${number}/label_right.nii.gz"

	mkdir -p "${SAVE}/nonLinear/case_00${number}"
	mkdir -p "${SAVE}/withLinear/case_00${number}"


	echo $leftImageData
	echo $rightImageData
	echo $leftImageNonLinearSave
	echo $rightImageNonLinearSave
	echo $leftImageWithLinearSave
	echo $rightImageWithLinearSave
	echo $leftLabelNonLinearSave
	echo $rightLabelNonLinearSave
	echo $leftLabelWithLinearSave
	echo $rightLabelWithLinearSave

	../deedsBCV/linearBCV -F $leftImageData -M $rightImageData -O right
	../deedsBCV/linearBCV -F $rightImageData -M $leftImageData -O left

	../deedsBCV/deedsBCV -F $leftImageData -M $rightImageData -O right_nonLinear
	../deedsBCV/deedsBCV -F $leftImageData -M $rightImageData -A right_matrix.txt -O right_withLinear -S $rightLabelData

	../deedsBCV/deedsBCV -F $rightImageData -M $leftImageData -O left_nonLinear
	../deedsBCV/deedsBCV -F $rightImageData -M $leftImageData -A left_matrix.txt -O left_withLinear -S $leftLabelData

	mv right_nonLinear_deformed.nii.gz $rightImageNonLinearSave
	mv right_withLinear_deformed.nii.gz $rightImageWithLinearSave
	mv left_nonLinear_deformed.nii.gz $leftImageNonLinearSave
	mv left_withLinear_deformed.nii.gz $leftImageWithLinearSave
	mv right_nonLinear_deformed_seg.nii.gz $rightLabelNonLinearSave
	mv right_withLinear_deformed_seg.nii.gz $rightLabelWithLinearSave
	mv left_nonLinear_deformed_seg.nii.gz $leftLabelNonLinearSave
	mv left_withLinear_deformed_seg.nii.gz $leftLabelWithLinearSave

	if [ $? -eq 0 ]; then
		echo "case_00${number} done."
	
	else
		echo "case_00${number}" >> fail/deeds.txt
		echo "case_00${number} failed"
	
	fi
	    

done

rm right*
rm left*
