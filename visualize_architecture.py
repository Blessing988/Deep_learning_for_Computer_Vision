#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import the ncessary packages 
from pyimagesearch.nn.conv.lenet import LeNet
from keras.utils import plot_model
import argparse
import pydot 
import graphviz


# In[ ]:


# construct the argument parse and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--architectures", required=True, help="path to images of network architectures")
args = vars(ap.parse_args())


# In[ ]:


# initialize LeNet and then write the network architecture
# visualization graph to disk
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file=args["architectures"], show_shapes=True)

