# Using Functional API - CNN
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import plot_model

visible = Input(shape=(256, 256, 3))
conv1 = Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu')(visible)
pool1 = MaxPooling2D(pool_size = (2,2))(conv1)
conv2 = Conv2D(16, kernel_size = 3, activation = 'relu')(pool1)
pool2 = MaxPooling2D(pool_size = (2,2))(conv2)
hidden1 = Dense(10, activation = 'relu')(pool2)
output = Dense(1, activation = 'sigmoid')(hidden1)
model = Model(inputs = visible, outputs = output)

print model.summary()
plot_model(model, to_file = '3_CNN.png')
