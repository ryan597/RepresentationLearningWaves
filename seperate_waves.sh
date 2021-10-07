#!/bin/bash

new_wave () {
	# recursively search for sequential images
	local I1=$1
	for i in {1..200}  # wave images should be within 2 seconds of eachother
	do
		local I2=$(($I1 + $i))

		if [ -e "$path/$I2.jpg" ]; then
			new_wave $I2 $index
			ln -s "$path/$I2.jpg" "$path/wave_$index/$I2.jpg"
			break
		fi
	done
}

###############################################################################

# Start with wave_0
index=0
total_count=0
path="$1"
# clean up from previous run
if [ -e $path/wave_1 ]; then
    rm -r $path/wave_*
fi

for image in $path/*.jpg
do
	I1=$(basename $image .jpg)   # remove path and .jpg extension
	# if this image is not in previous wave, create new wave
	if [ ! -e "$path/wave_$index/$I1.jpg" ]; then
		((index++))
		mkdir -p "$path/wave_$index"
		new_wave $I1 # search for all connecting images
		ln -s "$path/$I1.jpg" "$path/wave_$index/$I1.jpg"
		count=$(ls $path/wave_$index/*.jpg | wc -l)
		if [ $count != 0 ]; then
			echo -e "Images in wave_$index:\t $count"
			total_count=$(($total_count + $count))
		else
			# if there are no images in the wave delete the folder
			rm -r $path/wave_$index
			((index--))
		fi
	fi
done
echo "Total images processed: $total_count"
