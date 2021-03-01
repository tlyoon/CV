import numpy as np
import matplotlib.pylab as plt
import os
import pandas as pd
import plotly.express as px

sff='asc'
aors='ts'

### detect *.asc files
countfn=0;fn={}
for i in os.listdir():
    suffix=i.split('.')[-1]
    if i.split('.')[-1]==sff:
        fn[countfn]=i.split('.')[:-1][0]
#        print(countfn,fn[countfn],suffix)
        countfn=countfn+1

fn=[ i+'.'+sff for i in fn.values() ]
### import the *.npy data file saved and visualise it with imshow
#fig = plt.figure()
#ax3 = fig.add_subplot(121)
#ax3.set_title(aors)
#fo = open('G010-20C.asc', "r")
#newlist = [line.rstrip() for line in fo.readlines()]
#fo.close()

count=0;c2=0;
fo=open(fn[0], "r+")
for i in fo.readlines():
    count=count+1
    if i.strip()=='#DATA':
        id=count
fo.close()
print('id,count:',id,count)

l=[(None,None)]*count
plt.figure(figsize=(10,5))
title='JXL'
plt.title(title)
plt.ylabel('Normalized absorbance')
plt.xlabel('Wavelength'+r" (cm$^{-1}$)")


for ii in fn:
    count=0;c2=0;apair=[]
    fo=open(ii, "r+")
    for i in fo.readlines():
#        l[count]=[ line.strip() for line in i ]
#        print('count,l[count]:',count,l[count])
        if count > id:
            l[count]=i
#            print(ii,l[count])
            item=i.split();
            pair=[ float(item[i]) for i in range(len(item)) ]
            apair.append(pair)
            #print('pair:',pair)
        count=count+1
    print(ii,id)
    x=[ i[0] for i in apair ]
    y=[ i[1] for i in apair ]
    lgd=ii.split(".")[0][-3:]
    plt.plot(x, y, label = lgd) 
    fo.close()
plt.legend() 
plt.show()
#plt.savefig(title+'.jpg')
plt.savefig('{}.png'.format(title))


'''
df = pd.read_csv("G010-100C.csv")
fig = px.line(df, x = 'AAPL_x', y = 'AAPL_y', title='Apple Share Prices over time (2014)')
fig.show()
'''


'''
for i in range(id,count):
    print([ float(j) for j in l[i].split() ])
'''    
      

'''
print(data)
data=data[1:]
ax3.imshow(data, interpolation='nearest', cmap='jet')

ax3 = fig.add_subplot(122)
ax3.set_title('ta')
data = np.load("ta.npy")
data=data[1:]
ax3.imshow(data, interpolation='nearest', cmap='jet')
plt.show()
'''