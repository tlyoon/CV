### How to use this script. 
# Put all oversampled *.npy files of the fingerprints, e.g.,

# jinxianlian_ovsp.npy    shanyao_ovsp.npy
# gouwen_ovsp.npy         Mutong_ovsp.npy       zhexie_ovsp.npy

# in the same folder with this script. The script, upon execution, will decompose
# all individual fingerprin into a common folder decomposed/, in which the resultant files
# are in the form e.g.,
#gouwen.313.npy  jinxianlian.313.npy  Mutong.313.npy  shanyao.313.npy  zhexie.313.npy
#gouwen.314.npy  jinxianlian.314.npy  Mutong.314.npy  shanyao.314.npy  zhexie.314.npy
#gouwen.315.npy  jinxianlian.315.npy  Mutong.315.npy  shanyao.315.npy  zhexie.315.npy


import numpy as np
import os

## generate classes directory
#classes=[]
ovspfiles=[]
for i in os.listdir():
        try:
                cl=str(i).split('_ovsp.npy')[1]
                cl=cl=str(i).split('_ovsp.npy')[0]
                #print('cl:',cl)
#                classes.append(cl)
                ovspfiles.append(i)
        except:
                dummy=0
#classes=list(set(classes))
#print(classes)
#print(ovspfiles)
## end of generate classes directory
try:
    os.mkdir('decomposed')
except:
    dummy=0

for i in ovspfiles:
    clss=str(i).split('_ovsp.npy')[0]
#    print('reading',i,'for',clss)
    data=np.load(i)
    datalength=data.shape[0]
    print('datalength:',datalength)
    for j in range( datalength):
        fn=clss+'.'+str(j)+'.npy'
        #print(fn)
        np.save(os.path.join('decomposed',fn),data[j,:,:])
        #print('Exported',data[j,:,:] ,'into',fn)
        print(fn,'saved')
        #data[j,:,:]
    print('')