import numpy as np
import matplotlib.pylab as plt

### import the *.npy data file saved and visualise it with imshow
fig = plt.figure()
ax3 = fig.add_subplot(121)
ax3.set_title('ts')
data = np.load("ts.npy")
data=data[1:]
ax3.imshow(data, interpolation='nearest', cmap='jet')

ax3 = fig.add_subplot(122)
ax3.set_title('ta')
data = np.load("ta.npy")
data=data[1:]
ax3.imshow(data, interpolation='nearest', cmap='jet')


plt.show()