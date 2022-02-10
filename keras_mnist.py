#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import the necessary packages 
from pyimagesearch.nn.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential 
from keras.layers.core import Dense 
from keras.optimizers import SGD
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt 
import numpy as np
import argparse 


# In[ ]:


# construct the argument parse and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())


# In[4]:


#grab the MNIST dataset (if this is your time running this script, 
# the downlaod may take a minute -- the 55MB MNIST dataset will be downloaded)
print("[INFO] loading MNIST (full) dataset..")
dataset = fetch_openml('mnist_784')

#scale the raw pixel intensities to the range [0, 1.0], then 
# construct the training and testing splits
data = dataset.data.astype("float")/255.0
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25, random_state=42)

#Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY =  lb.fit_transform(testY)


# In[ ]:


# define the 784-256-128-10 architecture using keras 
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))


# In[ ]:


# training the model using SGD
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, 
              metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), 
              epochs=100, batch_size=128)


# In[ ]:


# evaluate the network 
print("[INFO] evaluating network")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), 
                            target_names=[str(x) for x in lb.classes_]))


# In[ ]:


#plot training loss and accuracy 
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])

