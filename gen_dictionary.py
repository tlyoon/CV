import os
import numpy as np

def gen_dictionary():
    pwd=os.getcwd()
    listdir=os.listdir(pwd)
#    print('listdir=',listdir)
    classdir=[]
    
    for cl in listdir:
        if os.path.isdir(cl) and cl != 'storage' and \
        cl != 'training_ovs_cache' and \
        cl != 'inference_ovs_cache' and \
        cl != 'inference' and \
        cl != 'png' and \
        cl != 'training' and \
        cl != 'record' and \
        cl != 'voting' and \
        cl != 'select_bpcls' and \
        cl != 'application' and \
        cl != '.ipynb_checkpoints' and cl != 'storage' and \
        cl != 'saved_model' and cl != '__pycache__' and cl != 'temp':
            #print (cl,os.path.isdir(cl))
            classdir=np.append(classdir,cl)
    
    d={}
    for i in range(len(classdir)):
        d[classdir[i]]=i
    
    nd = dict([(value, key) for key, value in d.items()]) 
    #print('dictionary for the classes =',d)
    savedict = open('dictionary.txt', 'w')
    savedict.write(str(d)+'\n')
    savedict.write(str(nd))
    savedict.close()
    
    data=open('dictionary.txt','r')
    savedict = open('dictionary.py', 'w')
    line=data.readline()    
    line2='dict='+line
    savedict.write(line2)
    line=data.readline()    
    line3='dict2='+line
    savedict.write(line3)
    data.close()   
    os.remove('dictionary.txt')
    return d
### end of def ###
####################################################################################

gen_dictionary()
