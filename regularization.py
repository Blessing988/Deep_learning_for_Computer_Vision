#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the necessary packages 
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse 


# The `SGDClassifier` encapsulates all the following concepts:
# * Loss functions
# * Number of epochs
# * Learning rate
# * Regularization terms
# 
# it makes it a perfect example to demonstrate all these concepts in action 

# Next, we can parse our `command line arguments` and grab the list of images from disk:

# In[ ]:


# construct the argument parse and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
                help="path to input dataset")
args = vars(ap.parse_args())

#grab the list of image paths 
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))


# Given the image paths, we'll resize to 32x 32 pixels, load them from disk into memory, 
# and then flatten into a 3027-dim array: 

# In[ ]:


# initialize the image preprocessor, load the dataset from the disk, and reshape the data matrix 
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0]), 3072)


# Encoding the labels as integers and performing a training testing split, using 75% of the data for 
# training and the remaining 25% for testing:

# In[ ]:


# encode the labels as integers 
le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)


# Applying a few different types of regularization when training our SGDClassifier

# In[ ]:


# loop over our set of regularizers
for r in (None, "l1", "l2"):
    
    print("[INFO] training model with {} penalty".format(r))
    model = SGDClassifier(loss="log", penalty=r, max_iter=100, learning_rate="constant", 
                          eta0=0.01, random_state=42)
    model.fit(trainX, trainY)
    
    #evaluate the classifier
    acc = model.score(testX, testY)
    print("[INFO] {} penalty accuracy: {:.2f}%".format(r, acc*100))

