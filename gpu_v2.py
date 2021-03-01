from __future__ import division
from numba import cuda
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import sys
import numba
from numpy import array

## subroutine to reduce array
def scale_down_arr(inp_arr):   
    arrayw=inp_arr
    rows=np.shape(arrayw)[0]
    columns=np.shape(arrayw)[1]

    if rows%2==0: ##if rows is even
        rsmax=int(rows/2)
    else: ##if rows is odd
        rsmax=int(rows/2)
       
    if columns%2==0: ##if columns is even
        csmax=int(columns/2)
    else:
        csmax=int(columns/2)                    
        
    array_sc= [[0 for x in range(csmax)] for y in range(rsmax)]
        #print(array)    
    count=0    
    for rs in range(0,rsmax):
        for cs in range(0,csmax):
            r=min(int(rs*2),rows)
            c=min(int(cs*2),columns)
            #print(rs,cs,r,r+2,c,c+2)
            #print(array[r:r+2,c:c+2])
            array22=arrayw[r:r+2,c:c+2]
            mean_array22=array22.mean()
            array_sc[rs][cs]=mean_array22
            #print('')
            count=count+1
            #print(columns,rows,count)
    array_sc=np.array(array_sc)
    out_array=array_sc
    return out_array

### to scale down inp_arr by a factor or 2^nscale 
def exponential_scale_down_arr(inp_arr,nscale): ###nscale >= 0
    scaled_arr=inp_arr
    for i in range(0,nscale):
        scaled_arr=scale_down_arr(scaled_arr)
        #print('i=',i)
    out_array=scaled_arr    
    return out_array
	

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


#### normalization     
def matrix_normalization(squarematrix):
    lenmatrix=len(squarematrix)
    xmax, xmin = squarematrix.max(), squarematrix.min()
    norm=[[ (squarematrix[ix][iy] -xmin)/(xmax - xmin) for iy in range(lenmatrix) ] for ix in range(lenmatrix) ]
    return norm


def gpu(files, place):
    m = len(files)
    #Stored the files into array
    output = []
    t = 1
    temp =[]
    for file in files:
        arr = []
        for line in file:
            if (len(line.split()) == 2 and isfloat(line.split()[0]) and isfloat(line.split()[1])):
                arr.append(float(line.split()[1]))
            if(len(line.split()) == 2 and t == 1 and isfloat(line.split()[0]) and isfloat(line.split()[1])):
                temp.append(float(line.split()[0]))
        t = 0
        output.append(arr)
#    print('len(temp)=',len(temp))
    output = np.array(output)

    threadsperblock = (8, 8)
    blockspergrid = (452, 452)

    ############################################
    #Synchronous
    #summation of output array
    SUM = np.sum(output, axis=0)
    #division of array
    DIVIDE = np.divide(SUM, m)

    #A_tudor array
    A_tudor = []
    for file in output:
        A_tudor.append(np.subtract(file, DIVIDE))
    A_tudor = np.array(A_tudor)

    # CUDA kernel
    @cuda.jit
    def multi(A,C):
        row, col = cuda.grid(2)
        z = row*3601+col
        if(z < 3601*3601):
            C[z] = A[row]*A[col]

    multi_A_bar = []
    
    for i in range(m):
        A = np.array(A_tudor[i])
        A_global_mem = cuda.to_device(A)
        C_global_mem = cuda.device_array(len(A_tudor[0])*len(A_tudor[0]))
        multi[blockspergrid, threadsperblock](A_global_mem,  C_global_mem)
        C = C_global_mem.copy_to_host()
        multi_A_bar.append(C)
        
    multi_A_bar_sum = np.sum(multi_A_bar, axis = 0)
    synchronous = np.divide(multi_A_bar_sum, m-1)

    ####################################
    #Asynchronous

    # CUDA kernel
    @cuda.jit
    def multi2(A,C):
        row, col = cuda.grid(2)
        z = row*3601+col
        C[z] = 0
        if(z < 3601*3601 and C[z] == 0):
            Sum = 0
            for i in range(0,m):
                v = 0
                for k in range(0,m):
                    if(k == i):
                       Nik = 0
                       v = v + Nik*A[k][row]
                    else:   
                       Nik = 1/(np.pi*(k-i))
                       v = v + Nik*A[k][row]
                Sum = Sum + A[i][col]*v
                   
            C[z] = Sum/(m-1)


    #A = np.array(A_tudor,dtype = 'f')
    A = np.array(A_tudor)
    A_global_mem = cuda.to_device(A)
    C_global_mem = cuda.device_array(len(A_tudor[0])*len(A_tudor[0]))
    multi2[blockspergrid, threadsperblock](A_global_mem,  C_global_mem)
    C = C_global_mem.copy_to_host()
    asyn = C
    
    #################################
    
    asyn_arr = [[0 for _ in range(len(temp))] for _ in range(len(temp))]
    synchronous_arr = [[0 for _ in range(len(temp))] for _ in range(len(temp))]

    z = 0
    for o in range(len(temp)): 
        for p in range(len(temp)):
            asyn_arr[p][o] = asyn[z]   
            synchronous_arr[p][o] = synchronous[z]
            z = z + 1
   
    asyn_arr = np.array(asyn_arr,dtype=float)
    synchronous_arr = np.array(synchronous_arr,dtype=float)

### call the scaling subroutine to scale input array	
    asyn_arr=exponential_scale_down_arr(asyn_arr,4)
    synchronous_arr=exponential_scale_down_arr(synchronous_arr,4)
    asyn_arr = np.array(asyn_arr)
    synchronous_arr = np.array(synchronous_arr)
    
### normalizing the 2D matrices
    Norm_asyn_arr=matrix_normalization(asyn_arr)
    Norm_synchronous_arr = np.array(synchronous_arr)
    
#    print('np.shape(asyn_arr)',np.shape(asyn_arr))
	
#    x = np.arange(0, 3601, 1)
#    y = np.arange(0, 3601, 1)
#    X, Y = np.meshgrid(x, y)
#    fig, ax = plt.subplots(figsize=(6,6))
#    ax.contour(X, Y, synchronous_arr)
    fig, axsyn = plt.subplots(figsize=(6,6))
    axsyn.imshow(synchronous_arr, interpolation='nearest', cmap='jet')
    plt.savefig(place + "/" + "ts.png")
  
    
#    x = np.arange(0, 3601, 1)
#    y = np.arange(0, 3601, 1)
#    X, Y = np.meshgrid(x, y)
#    fig, ax = plt.subplots(figsize=(6,6))
#    ax.contour(X, Y, asyn_arr)
    fig, axasyn = plt.subplots(figsize=(6,6))
    #plt.imshow(asyn_arr)
    axasyn.imshow(asyn_arr, interpolation='nearest', cmap='jet')
    plt.savefig(place + "/" + "ta.png")
    plt.close('all')
	
#    np.save(place + "/"+"ta.npy", asyn_arr)
#    np.save(place + "/"+"ts.npy", synchronous_arr)

    np.save(place + "/"+"ta.npy", Norm_asyn_arr)
    np.save(place + "/"+"ts.npy", Norm_synchronous_arr)


##############################################   
'''
    aaa = array(asyn_arr,'float32')
    output_file = open(place + "/" +'ta.dat', 'wb')
    aaa.tofile(output_file)
    print('place + "/" + ta.dat',place + "/" +'ta.dat')
    output_file.close()

    aaa = array(synchronous_arr,'float32')
    output_file = open(place + "/" +'ts.dat', 'wb')
    aaa.tofile(output_file)
    print('place + "/" + ts.dat = ',place + "/" +'ta.dat')
    output_file.close()
'''    

##############################################   
    #doa=np.shape(asyn_arr)[0] 
    #hash=[["#" for i in range(doa)]]
    #asyn_arr=np.concatenate((hash, asyn_arr.T), axis=0)
    #synchronous_arr=np.concatenate((hash, synchronous_arr.T), axis=0)
##############################################
####################################

'''
    #Write synchronous and asynchronous file
    s = open(place+"/ts.dat", "w")
    a = open(place+"/ta.dat", "w")
    s.write('#'+"\n")
#    s.write("\n")
    a.write('#'+"\n")
#    a.write("\n")	
   
#    z = 0
#    for o in range(len(temp)):
#        for p in range(len(temp)):
#            s.write(str(temp[o]) + "   " + str(temp[p]) + "   " + str(synchronous[z])+"\n")
#            #s.write("\n")
#            a.write(str(temp[o]) + "   " + str(temp[p]) + "   " + str(asyn[z])+"\n")
#            #a.write("\n")
#            z = z + 1

    for o in range(len(asyn_arr)):
         acolumn=[ asyn_arr[o][p] for p in range(len(asyn_arr)) ]
         scolumn=[ synchronous_arr[o][p] for p in range(len(synchronous_arr)) ]
         s.write(str(scolumn))
         s.write("\n")
         a.write(str(acolumn))
         a.write("\n")		
    
    s.close()
    a.close()
    
'''       


#cuda.current_context().reset()

arr = os.listdir()
arr.sort()

#item_store = []
#for item in arr:
#   if(item != "gpu.py" and item != "nohup.out" and item !=  "edward.py"):
#      item_store.append(item)
#
#arr = item_store

########################
item_store=[]
p=os.listdir()
for i in p:
    if  os.path.isdir(i):
        #print(i)
        item_store.append(i)
arr = item_store		
print('print from gpu.py, found ',len(item_store),' directories:',arr)
#############################


def gpu_file(xxx, yyy, zzz):
   total = 0
   files = []
#   place = os.path.abspath(__file__)[:]
   place = os.getcwd()
#   print('place=',place)
#[0:-6]
   
   for item in arr[zzz-1:zzz]:
#      items = os.listdir(place + item)
      items = os.listdir(os.path.join(place,item))
      #print('items = ',items)
#	  items = os.path.join( os.getcwd(),item)
      items.sort()
      #print(items)
      for i in items[xxx:yyy]:
#         p = os.listdir(place + item + "/" +i)
         p = os.listdir(os.path.join(place,item,i))
         for a in p:
            if("asc" in a):
               total = total + 1
#               pl = place + item + "/" + i + "/" + a
               pl = os.path.join(place,item,i,a)
#               print('****',pl)
               f = open(pl, "r")
               files.append(f)
            
         if(total > 0):
#             print("print from gpu_v2.py: working in ",place + item + "/" + i)
             print("print from gpu_v2.py: working in ",os.path.join(place,item,i))
#             gpu(files, place + item + "/" + i)
             gpu(files,os.path.join(place,item,i))
             files = []
             total = 0
    
if __name__== "__main__":
    
    #print(sys.argv[3])
    gpu_file(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    #print(0, 20, 1)



#numba.cuda.profile_stop()
#gpu_file(0, 20, 1)
#numba.cuda.profile_stop()
#export NUMBAPRO_NVVM=/usr/local/cuda-8.0/nvvm/lib64/libnvvm.so.3.1.0
#export NUMBAPRO_LIBDEVICE=/usr/local/cuda-8.0/nvvm/libdevice

