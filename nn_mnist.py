#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages 
from pyimagesearch.nn.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from sklearn import datasets 


# In[7]:


#load the MNINST dataset and apply min/max scaling to scale the pixel intensity 
#values to the range [0, 1] (each image is
#represented by an 8 x 8 = 64-dim feature vector
print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))


# In[12]:


# construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

#convert the labels from integers to vector
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)


# In[19]:


# train the network 
print("[INFO] training network...")
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY, epochs=1000)


# In[20]:


# evaluate the network
print("[INFO] evaluating network...")
predictions = nn.predict(testX)
predictions = predictions.argmax(1)
print(classification_report(testY.argmax(axis=1), predictions))


# In[15]:





# In[18]:


digits.target


# In[ ]:




