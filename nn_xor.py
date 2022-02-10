#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the necessary packages 
from pyimagesearch.nn.neuralnetwork import NeuralNetwork
import numpy as np

#construct the XOR Dataset 
X = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

#define our neural network architecture

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000)

# loop over our XOR data points 

for (x, target) in zip(X, y):
    #make prediction our data point and display the result to our console
    pred = nn.predict(x)[0][0]
    step=1 if pred>0.5 else 0
    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))

