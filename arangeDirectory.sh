#!bin/bash

#Input
readonly DATA="$HOME/Desktop/data/box/OBB"
readonly SAVE="$HOME/Desktop/data/box/OBB"

numArr=(000 001 003 004 006 007 009 010 014 015 017 018 019 020 022 023 027 031 032 033 037 039 040 043 049 050 052 054 062 063 064 065 071 072 075 076 077 081 082 083 085 091 093 094 096 097 100 101 103 106 115 120 121 123 124 125 127 128 129 132 136 137 138 140 141 146 150 152 153 155 156 158 164 167 173 174 175 182 188 190 191 193 198 201 203 205 )

for number in ${numArr[@]}
do
	data="${DATA}/case_00${number}"
	original="${SAVE}/original"
	transform="${SAVE}/transform"
	
	originalSave="${original}/case_00${number}"
	transformSave="${transform}/case_00${number}"


	echo $data
	echo $originalSave
	echo $transformSave

	mkdir -p $originalSave
	mkdir -p $transformSave

	cp "${data}/image_left.nii.gz" "${originalSave}/image_left.nii.gz"
	cp "${data}/image_right.nii.gz" "${originalSave}/image_right.nii.gz"
	cp "${data}/image_left_transformed.nii.gz" "${transformSave}/image_left.nii.gz"
	cp "${data}/image_right_transformed.nii.gz" "${transformSave}/image_right.nii.gz"
	cp "${data}/label_left.nii.gz" "${originalSave}/label_left.nii.gz"
	cp "${data}/label_right.nii.gz" "${originalSave}/label_right.nii.gz"
	cp "${data}/label_left_transformed.nii.gz" "${transformSave}/label_left.nii.gz"
	cp "${data}/label_right_transformed.nii.gz" "${transformSave}/label_right.nii.gz"

done
