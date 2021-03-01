#! /bin/bash

for i in $(ls)
do
	if [ ! "$i" = 'unlink_all.sh' ]; then
	echo 'unlink ' $i
	unlink $i
	fi
done
