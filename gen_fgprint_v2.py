#from subprocess import Popen
import os

place = os.path.abspath(os.getcwd())

arr = os.listdir(place)
arr.sort()

########################
item_store=[]
p=os.listdir(place)
for i in p:
    if  os.path.isdir(i):
        #print(i)
        item_store.append(i)
#print('Found ',len(item_store),' directories:',item_store)
#############################

arr = item_store

o = 1

size = 10
for item in arr[:]:
#   print('item=',item)
   #items = os.path.abspath(__file__) + item
   #classdir = os.path.abspath(__file__)
   items = os.path.join( os.getcwd(),item)
   ##
#   classdir=os.path.abspath(i)
#   print('classdir=',classdir)
   ##
#   print('path of class directory ',items)
   big = len(os.listdir(items))
#   print('big=',big)
   s = int(big/size + (1 - (big/size - int(big/size))))
#   print('s=',s)
   #for i in range(s):
   for i in range(s):
      if(big <= (i+1)*size):
         os.system("python gpu_v2.py " + str(i*size) +" " + str(i*size + (size - ((i+1)*size - big) )) + " " + str(o))
      else:
         os.system("python gpu_v2.py " + str(i*size) +" " + str((i+1)*size) + " " + str(o))
   
   o = o  + 1
