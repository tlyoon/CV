#! /bin/bash

count=0
countj=0
countpng=0
for i in $(ls -d */)
	do
	cd $i
	#echo $PWD
	dir=$(ls -d */)
#	echo $PWD
#	echo $dir
	for j in $dir
	do	
#		echo $j
		cd $j
        countj=$(( $countj + 1 ))
#		if [ -f ta.png ]; then 
        if [ -f ta.npy ] && [ -f ts.npy ]; then
		#    echo 'j=' $j
			#ls -la t*.dat
			#echo both ta.dat and ts.dat are found in $PWD
            countnpg=$(( $countnpg + 1 ))
			echo $i$j has both ts.npy and ta.npy
		else
			#echo $PWD has no ts.npy or ta.npy
			count=$(( $count + 1 ))
            dwodat[$count]=$i$j
			echo ${dwodat[$count]} has no ts.npy or ta.npy
		fi
#		ls
		cd ../
	done
	cd ../
	done
echo 'No. of expected ta.npy (and ts.npy) = ' $countj
echo 'No. of ta.npy (and ta.npy) found = ' $countnpg
echo 'No. of directories without ta.npy or ts.npy =' $count. 
#echo 'They are ${dwodat[@]}
