#!bin/bash

#Input
readonly DATA="$HOME/Desktop/data"
readonly SAVE="$HOME/Desktop/data/watch"

testing=(019 023 054 093 096 123 127 136 141 153 188 191 201)
for number in ${testing[@]}
do
	originalCT="${DATA}/kits19/case_00${number}/imaging.nii.gz"
	originalLabel="${DATA}/kits19/case_00${number}/segmentation.nii.gz"
	segmentationLabel="${DATA}/slice/newOriginal/segmentation/case_00${number}/label.mha"
	
	save="${SAVE}/case_00${number}"
	
	mkdir -p $save
	cp $originalCT "${save}/imaging.nii.gz"
	cp $originalLabel "${save}/label.nii.gz"
	cp $segmentationLabel "${save}/label_seg.mha"

done
