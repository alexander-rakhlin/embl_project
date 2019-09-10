import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from time import time
import sompy

dlen = 700
tetha = np.random.uniform(low=0,high=2*np.pi,size=dlen)[:,np.newaxis]
X1 = 3*np.cos(tetha)+ 1.22*np.random.rand(dlen,1)
Y1 = 3*np.sin(tetha)+ 1.22*np.random.rand(dlen,1)
Data = np.concatenate((X1,Y1),axis=1)

mapsize = 50
# this will use the default parameters, but i can change the initialization and neighborhood methods
som = sompy.SOMFactory.build(Data, mapsize, mask=None, mapshape='cylinder', lattice='rect',
                             normalization=None, initialization='spherical', neighborhood='gaussian',
                             training='batch', name='sompy', track_history=True)
som.train(n_job=1, verbose=None)  # verbose='debug' will print more, and verbose=None wont print anything


step = -1
# som_matrix = som.codebook.matrix
# som_bmu = som._bmu[0]
som_matrix = som.history[step][0]
som_bmu = som.history[step][1]

fig = plt.figure()
plt.scatter(Data[:,0], Data[:,1], c=np.abs(som_bmu - mapsize / 2))
plt.plot(som_matrix[:,0], som_matrix[:,1], '-', markersize=4)
plt.show()

som.predict()