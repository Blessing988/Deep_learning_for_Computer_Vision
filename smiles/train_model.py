#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagesearch.nn.conv.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import imutils 
import cv2
import os 


# In[ ]:


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())


# In[2]:


# initialize the list of data and labels 
data = []
labels = []


# In[ ]:


# loop over the input images 
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)
    
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label=="positives" else "not_smiling"
    labels.append(label)


# In[ ]:


# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)


# In[ ]:


# Handling data imbalance
# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
classWeight = {0:classWeight[1], 
               1:classWeight[0]}

# In[ ]:


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, 
      labels, test_size=0.20, stratify=labels, random_state=42)


# In[ ]:


# initialize our model 
print("[INFO] compiling the model...")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", 
              metrics=["accuracy"])


# In[ ]:


# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), 
      class_weight=classWeight, batch_size=64, epochs=15, verbose=1)


# In[ ]:


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), 
        predictions.argmax(axis=1), target_names=le.classes_))


# In[ ]:


# save the model 
model.save(args["model"])


# In[ ]:


#plot training + validation loss and accuracy 
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

