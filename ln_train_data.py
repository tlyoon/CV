import os

st="cd training; classes=$(ls -d */ | sed 's/[storage/]//g' | awk 'NF==1 {print}'); echo $classes > ../classes.dat"
#print(st)

os.system(st)

st="for i in $(cat classes.dat); do ln -s training/$i . ; done; rm classes.dat"
os.system(st)


