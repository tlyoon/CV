import os, shutil
pwd=os.getcwd()
path=os.path.join(pwd,'png')
#print('path 1:', path)
if not(os.path.isdir(path)):
	os.mkdir(path)
#print('path 2:',path)


sourcepath=pwd
sourcefiles = os.listdir(sourcepath)
destinationpath=path
#sourcepath='C:/Users/kevinconnell/Desktop/Test_Folder/'
#destinationpath = 'C:/Users/kevinconnell/Desktop/Test_Folder/Archive'
for file in sourcefiles:
    if file.endswith('.png'):
        shutil.move(os.path.join(sourcepath,file), os.path.join(destinationpath,file))
