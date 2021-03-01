import os, shutil

##
## generate classes directory
collect=[]
for i in os.listdir():
        try:
                suffix=str(i).split('.')[2]
                if suffix=='npy':
#                        print(i)
                        title=str(i).split('.')[0]
                        collect.append(title)
        except:
                dummy=0
classes=list(set(collect))
#print(classes)
##
pwd=os.getcwd()
for cls in classes:
    if (os.path.isdir(cls)):
        shutil.rmtree(cls)

for cls in classes:
    if (not os.path.isdir(cls)):
        os.mkdir(cls)
        
    for i in os.listdir():
        if ((not (str(i) in classes)) and (str(i)!='oversample')):
            label=str(i).split('.')[1]
            if str(i).split('.')[0]==cls:
                os.mkdir(os.path.join(pwd,cls,label));state='mv '+os.path.join(pwd,i)+' ' + os.path.join(pwd,cls,label,'ta.npy')
#                print('state:',state)
                os.system(state)
                #os.symlink(os.path.join(pwd,i),os.path.join(pwd,cls,label,'ta.npy'))
                #print('see this?')print(' ')
os.chdir(pwd)
print('all *.npy in current directory have been moved into the respective class directory',classes)
##

#for cls in classes:
#    if (os.path.isdir(cls)):
#        print(cls)   
    #    shutil.rmtree(cls)
