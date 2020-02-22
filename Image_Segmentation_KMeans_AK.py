#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import imageio
from os import listdir,makedirs
from os.path import isfile,join
#from scipy import ndimage
#from scipy import misc
from sklearn import cluster
import numpy as np
import os
import os.path, sys
from PIL import Image

path = "Documents/University Work/SMU Masters/CS 7345 - Advanced Application Programming/Project/training-20200209T221005Z-001/training"

dirs = os.listdir(path)
for item in dirs:
    fullpath = os.path.join(path,item)
    if os.path.isfile(fullpath):
        img = np.array(Image.open(fullpath))
       # plt.figure(figsize = (15,8))
        #plt.imshow(img)
        
        x, y, z = img.shape
        image_2d = img.reshape(x*y, z)
        image_2d.shape
        
        kmeans_cluster = cluster.KMeans(n_clusters=2)
        kmeans_cluster.fit(image_2d)
        cluster_centers = kmeans_cluster.cluster_centers_
        cluster_labels = kmeans_cluster.labels_
        cluster_centers[cluster_labels]
        
        plt.figure(figsize = (15,8))
        plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z).astype('uint8'))
        #plt


# In[3]:



  #image = imageio.imread(img)
  #image = ndimage.imread("dress.jpg")
  


# In[ ]:




