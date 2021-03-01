# /bin/bash

## for l in $(ls -d */); do cd $l; for i in  $(ls -d */); do  (for j in $i; do cd $j; for k in $(ls); do  cd $k; ( cwd=$PWD; ls "$PWD"/cvp.txt;  ssh comsics.usm.my "hostname; echo $cwd; mkdir -p $cwd"; scp "$PWD"/cvp.txt comsics.usm.my:$cwd; cd $cwd; ) ; cd ../; done  ;cd ../; done ) done ; cd ../; done
