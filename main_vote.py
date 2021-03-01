import os,sys,clfs
import numpy as np
from global_param import *

pwd=os.getcwd()
orig_stdout = sys.stdout

## The data of the fingerprints must have been contained in the directory training/GW/, training/SY/, ..., etc.
## Excute ln_train_data.py to create a link of the class subfolder e.g., GW/ in the current directory.
file2execute="ln_train_data.py"
fn=os.path.join(pwd,file2execute)
print('python script to execute:',fn)
exec(open(fn).read())



# =============================================================================
for iNround_evaluate in range(Nround_evaluate):
    np.save('iNround_evaluate',iNround_evaluate)
    classifiers=clfs.ccl()
    classifiers=[ str(i).split('()')[0] for i in classifiers ]
#     
    print('begin of a new iNround',iNround_evaluate)
    ### pre-process data    
    file2execute="data_preprocess_vote_v2.py"
    fn=os.path.join(pwd,file2execute)
    print('python script to execute:',fn)
    # To execute python fn from this sript
    exec(open(fn).read())
#     
    ### train models    
    file2execute="training_1vA.py"
    fn=os.path.join(pwd,file2execute)
    print('python script to execute:',fn)
    # To execute python fn from this sript
    exec(open(fn).read())
    print('')
# ### exit for iNround_evaluate loop


#### move all *.png file to png/
file2execute1="mvpng.py"
fn1=os.path.join(pwd,file2execute1)
print('To execute '+file2execute1)
exec(open(fn1).read())


### unlink classes folder 
st="cd training; classes=$(ls -d */ | sed 's/[storage/]//g' | awk 'NF==1 {print}'); echo $classes > ../classes.dat"
os.system(st)
st="for i in $(cat classes.dat); do unlink $i ; done; rm classes.dat"
os.system(st)
# =============================================================================

externalfile1="cvp.txt"
try: 
    os.remove(externalfile1)
except:
    nothing=0

#file2execute1="voting2.py"
file2execute1="cvp.py"
fn1=os.path.join(pwd,file2execute1)
print('To execute '+file2execute1+' and redirect it to '+externalfile1)
sys.stdout = open(externalfile1, "w")   #### comment this if want to print only to screen and dont want to redirect into *.txt
exec(open(fn1).read())
sys.stdout = orig_stdout
#print('sys.stdout after '+file2execute1,sys.stdout)
print('cvp.py finished')
print('')


