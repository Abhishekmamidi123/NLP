# Shared Input Layer model
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.utils import plot_model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

visible = Input(shape=(64,64,1))

conv1 = Conv2D(32, kernel_size = 4, activation = 'relu')(visible)
pool1 = MaxPooling2D(pool_size = (2,2))(conv1)
flat1 = Flatten()(pool1)

conv2 = Conv2D(16, kernel_size = 8, activation = 'relu')(visible)
pool2 = MaxPooling2D(pool_size = (2,2))(conv2)
flat2 = Flatten()(pool2)

merge = concatenate([flat1, flat2])
hidden1 = Dense(10, activation = 'relu')(merge)
output = Dense(1, activation = 'sigmoid')(hidden1)

model = Model(inputs = visible, outputs = output)
print model.summary()
plot_model(model, to_file = '5_Shared_input_layers.png')
