#!/usr/bin/env python
# coding: utf-8

# In[1]:


# set the background backend so figure can be saved in the background 
import matplotlib
matplotlib.use("Agg")

# import the necessary packages 
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os 


# In[ ]:


# construct the argument parse and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to the output directory")
args = vars(ap.parse_args())

# show information on the process ID
print("[INFO process ID:".format(os.getpid()))


# In[ ]:


# load the training and testing data, then scale it into the range [0, 1]
print("[INFO] loading Cifar-10 dataset...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)


# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# In[ ]:


# initialize the optimizer and model 
print("[INFO] compiling model...")
opt = SGD(lr= 0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, 
              metrics=["accuracy"])


# In[ ]:


# construct the set of callbacks 
figPath =  os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]


# In[ ]:


# train the network 
print("[INFO] training network...")
model.fit(trainX, trainY, validation_data=(testX, testY), 
          batch_size=64, epochs=100, callbacks=callbacks, verbose=1)

