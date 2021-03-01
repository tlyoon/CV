#! /bin/bash

dirs=$(ls -d */)
pwd=$PWD
for clsdir in $dirs
	do
	cd $clsdir
	localdir=$(ls -d */)
	for j in $localdir
		do
		#cd $j
		name=$(echo $j | awk -F"/" '{print $1}')
		#echo $name
#		echo $(pwd)
		n2=$(echo $clsdir | awk -F"/" '{print $1}')
#		echo 'n2:' $n2
		echo ln -s $n2/$j'ta.npy' $pwd/$n2.$name'.npy'
#		cp $j'ta.npy' ../$n2.$name'.npy'
		ln -s $n2/$j'ta.npy' $pwd/$n2.$name'.npy'
		
#		echo ''
		#cd ../
		done
#	echo $i/ta.npy zhexie.$name.npy
#	cp $i/ta.npy zhexie.$name.npy
	cd ../
	done
