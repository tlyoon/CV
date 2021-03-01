Nnofs=5       #### The number of oversample in each class is defined as nofs=Nnofs*maxldf
              #### Note that the number of original samples in each class in general is different. maxldf refers to the number of samples in the largest class)


##### don't touch anyting below ###################

import os
import numpy as np
from random import shuffle
from collections import Counter
#import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN #


#from sklearn.preprocessing import normalize
#from numpy import where
import imblearn
from imblearn.over_sampling import KMeansSMOTE
#from imblearn.datasets import make_imbalance
#from imblearn.pipeline import make_pipeline
#import pandas as pd

fracfake=1.0  #### 0.49 the default ratio for splitting the original data in the class/sample subdirectories
              #### of all classes into fake and authetic samples according to 
              #### Say if a class/ has 10 ta.npy files in its subdirectory, 
              #### 1 + int(fracfake*10) of them will be used as 'seed' for generating
              #### over-sampling (fake) data for training purpose, the rest, 10 - 1 + int(fracfake*10) will
              #### remain as 'authentic' data with out. The total number of oversampling
              #### is determined by the parameter Nnofs.
              
#X1=1;X2=10  ##### components in the high-dimensional data point to be displayed for visualisation

pwd=os.getcwd()
listdir=os.listdir(pwd)
#print('listdir=',listdir)

classdir=[];maxldf=-100  

for cl in listdir:
    if os.path.isdir(cl) and cl != 'storage' and \
    cl != '.ipynb_checkpoints' and cl != 'storage' and \
    cl != 'saved_model' and cl != '__pycache__':
        #print (cl,os.path.isdir(cl))
        classdir=np.append(classdir,cl)
print('classdir:',classdir)

#d={}
#for i in range(len(classdir)):
#    d[classdir[i]]=i

#print('dictionary for the classes =',d)

'''
#np.save("dictionary.dat",d)
file = open('dictionary.dat',"w")
file.write(d)
file.close()
'''
#savedict = open('dictionary.txt', 'w')
#savedict.write(str(d))
#savedict.close()


#kj=0; 
dirfake=[]
dirauth=[]

#### asyn ####
for i in range(len(classdir)):
    cl=classdir[i]
    dirinclass=os.listdir(cl)
    
    if '.ipynb_checkpoints' in dirinclass :
        dirinclass.remove('.ipynb_checkpoints')
    
    if 'saved_model' in dirinclass :
        dirinclass.remove('saved_model')
        
    if 'storage' in dirinclass :
        dirinclass.remove('storage')

    if '__pycache__' in dirinclass :
        dirinclass.remove('__pycache__')
    
    lendirinclass=len(dirinclass)
    
    dirinclass=[ os.path.join(cl,dirinclass[i],'ta.npy') for i in range(lendirinclass) ]
    
    print('i=',i,'cl=',cl, 'lendirinclass=',lendirinclass)
    #print('dirinclass=',dirinclass)
#    if lendirinclass==1:
#        print('only one sample fgp in ', cl);
#    else:
#    ldf=1+int(lendirinclass/2)
    ldf=int(fracfake*lendirinclass)
    print('total sample in class ',cl, 'is',lendirinclass )
    #print('to split', dirinclass, 'into auth and fake samples')
    print('no of fake samples is ',ldf)
    maxldf=max(ldf,maxldf)
#        print('no of auth samples is ',max(1,lendirinclass-ldf))
    shuffle(dirinclass)
    #print('after shuffle',dirinclass)
    dir2fake=dirinclass[:ldf]
#    print('dir2fake = ',dir2fake)
    dirfake.append(dir2fake)
#    print('dirfake=',dirfake)
    diff = set(dirinclass) - set(dir2fake)
    diff = list(diff)
#    print('after set difference, dirauth=',dirauth)
    if len(diff)==0:
        diff=dirinclass
        
#    if firsttime==True:
#        firsttime=False
#    else:
    dirauth.append(diff)
    
#    print('1 dirauth=',dirauth)
 #   print('dirauth=',dirauth)
    print('')

    
containerf=[];label_fake=[];y_fake=[];y_fake_int=[]
for i in range(len(dirfake)):
#for i in range(len(dirinclass)):
    print('')
    #print ('i,dirfake[i]',i,dirfake[i])
    for j in range(len(dirfake[i])):
        #npy=os.path.join(pwd,dirfake[i][j])
        npy=dirfake[i][j]
     #   print(j,npy,os.path.isfile(npy))
        if os.path.isfile(npy):
#                print('to append', npy,'into fake_a.npy. i =',i)
            datain=np.load(npy)
            ##################################
            containerf.append(datain)
            label_fake.append(npy+'; y='+ str(i))
#            print('label_fake= ',label_fake)
            y=npy.split(os.sep)[0]
            y_fake.append(y)
            y_fake_int.append(i)
            ##################################
#            print(container)
            #print(datain)

print('')
#print('maxldf=',maxldf)
nofs= Nnofs*maxldf      ##### The number of faked samples in each class required


containerf=np.array(containerf)
#np.save('fake_data_a.npy', containerf)
#np.save('fake_labels_a.npy', label_fake)
#np.save('fake_data_y_a.npy', y_fake)
#np.save('fake_data_y_int_a.npy', y_fake_int)
print('these are samples to be faked :',label_fake)
print(' ' )    

container=[];label_auth=[];y_auth=[];y_auth_int=[]

y_res_k=[]
cdict={}
for i in range(len(dirauth)):
    for j in range(len(dirauth[i])):
        npy=dirauth[i][j]
        if (os.path.isfile(npy)):
#            print('to append', npy,' into auth_a.npy')
            datain=np.load(npy)
            container.append(datain)
            label_auth.append(npy+'; y='+ str(i))
            y=npy.split(os.sep)[0]
            cdict[i]=y
#                print('i,y,npy',i,y,npy)
#                print('cdict[i],i:',cdict[i],i)
            y_auth.append(y)
            y_auth_int.append(i)
            ####
            dummy=[0 for k in range(len(classdir)) ]
            dummy[i]=1
#            print('dummy=',dummy)
            y_res_k.append(dummy)
            
#    print('these are samples to be kept authenic :',label_auth)

#np.save('auth_data_a.npy', container)
#np.save('auth_labels_a.npy', label_auth)
#np.save('auth_data_y_a.npy', y_auth)
#np.save('auth_data_y_int_a.npy', y_auth_int)
#np.save('auth_data_y_int_a_'+str(len(classdir))+'.npy', y_res_k)

print('')


####################################################
print('oversampling begins ')

k=len(classdir);seed=10

X=containerf
#    datalabels = label_fake
y = y_fake_int
#y = y_fake
#X = np.load("fake_data_a.npy")
#datalabels = np.load("fake_labels_a.npy")
#y = np.load("fake_data_y_a.npy")
#y = np.load("fake_data_y_int_a.npy")
#    counter = Counter(y)
#print("counter before oversampling = ",counter)
print("counter before oversampling = ", sorted(Counter(y).items()))


#print('1',X.shape)
X = np.reshape(X, (X.shape[0], X.shape[2]*X.shape[2]))
#print('2',X.shape)

####### scatter plot of X and y
#plt.xlabel('x')
#plt.ylabel('y')
#plt.scatter(X[:, X1], X[:, X2], marker='o', 
#               c=y, s=25, edgecolor='k', cmap=plt.cm.coolwarm)
#plt.show()


#### stating sampling_strategy #####
#sampling_strategy = {'gouwen': 60, 'Mutong': 60, 'jinxianlian': 60}
#sampling_strategy = {0: nofs, 1: int(nofs), 2: nofs}
lcdir=len(classdir)
sampling_strategy={}
for i in range(int(lcdir)):
    sampling_strategy[i]=nofs  
print('sampling_strategy = ',sampling_strategy)


##### implementing oversampling ####
X_res, y_res= SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k-1, random_state=seed).fit_resample(X, y)
#X_res, y_res=imblearn.over_sampling.BorderlineSMOTE(sampling_strategy=sampling_strategy,random_state=seed,k_neighbors=k).fit_resample(X, y)
#X_res, y_res = ADASYN(random_state=42,sampling_strategy=sampling_strategy,n_neighbors=k).fit_resample(X, y)
#### implementing oversampling ####

#plt.xlabel('x')
#plt.ylabel('y')
#plt.scatter(X_res[:, X1], X_res[:, X2], marker='o', 
#               c=y_res, s=25, edgecolor='k', cmap=plt.cm.coolwarm)
#plt.show()

y_res_3=[]
for i in range(len(y_res)):
    dummy=[0 for j in range(k) ]
    dummy[y_res[i]]=1
    #print(y_res[i],dummy)
    y_res_3.append(dummy)
    
print("counter after oversampling = ", sorted(Counter(y_res).items()))


dim=int(X_res.shape[1]**0.5)
X_res=X_res.reshape(X_res.shape[0],dim,dim)

### to work here
print('')
for j in classdir:
    countfn2s=0
    for i in range(len(y_res)):
#            print(i,y_res[i],cdict[y_res[i]])
        if cdict[y_res[i]]==j:
            countfn2s=countfn2s+1
            fn2s=j + '.' + str(countfn2s) + '.npy' 
#                print('file name to save:',fn2s)
#            print('To save item',i,'in X_res', 'as',fn2s,'for class',j)
            np.save(fn2s, X_res[i])
#    print('')

print('Counting number of members in each oversampled classes')
for i in range(len(classdir)):
    print(classdir[i],y_res.count(i))

### end of to work here 
#    np.save('ovsp_data_y_a.npy', y_res)
#    np.save('ovsp_data_a_'+str(k)+'.npy', X_res)
#    np.save('ovsp_data_y_a_'+str(k)+'.npy', y_res_3)

tnpd=[ Counter(y_res)[keys] for keys in Counter(y_res)][0]*len(Counter(y_res))
print('Total number of present data = ', tnpd)
tnod=[ Counter(y)[keys] for keys in Counter(y)][0]*len(Counter(y))
print('Total number of original data = ',tnod)
print('Ratio of fake data to original data = ',(tnpd/tnod))

### end of def ###
####################################################################################



