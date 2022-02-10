#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import the necessary packages 
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # Tensorflow backend only 
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2


# In[ ]:


# construct the argument parse and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, 
                help="path to input image")
ap.add_argument("-m", "--model", type=str, default="vgg16", 
                help="name of the pre-trained network to use")
args = vars(ap.parse_args())


# In[ ]:


# define a dictionary that maps model names to their classes inside Keras
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet" : ResNet50
}


# In[ ]:


# ensure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command argument should be a key in the 'MODELS' dictionary")


# In[ ]:


# initialize the input image shape (224 x 224 pixels) along with 
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)

inputShape = (224, 224)

preprocess = imagenet_utils.preprocess_input

# if we are using the InceptionV3 or Xception networks, then we
# need to set the input shape to (299x299) [rather than (224x224)]
# and use a different image processing function

if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input


# In[ ]:


# load our the network weights from disk (NOTE: if this is the
# first time you are running this script for a given network, the
# weights will need to be downloaded first -- depending on which
# network you are using, the weights can be 90-575MB
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")


# In[ ]:


# load the input image using the Keras helper utility while ensuring
# the image is resized to ‘inputShape‘, the required input dimensions
# for the ImageNet pre-trained network

print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)


# In[ ]:


# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through the network
image = np.expand_dims(image, axis=0)

# pre-process the image using the appropriate function based on the
# model that has been loaded (i.e., mean subtraction, scaling, etc.)
image = preprocess(image)


# In[ ]:


# classify the image 
print("[INFO] classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))


# In[ ]:


# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen
orig = cv2.imread(args["image"])
dim = (480, 300)
orig = cv2.resize(orig, dim, interpolation = cv2.INTER_AREA)
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label : {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)

