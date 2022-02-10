from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet5:

    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(
            6,
            kernel_size=(5, 5),
            strides=(1,1),
            padding="same",
            input_shape=inputShape
        ))
        model.add(Activation("tanh"))
        model.add(AveragePooling2D())
        model.add(Conv2D(
            16,
            strides=(1,1),
            kernel_size=(5, 5),
            padding="valid"
        ))
        model.add(Activation("tanh"))
        model.add(AveragePooling2D())
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dense(84))
        model.add(Activation("tanh"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
