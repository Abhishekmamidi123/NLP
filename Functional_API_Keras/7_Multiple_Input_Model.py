# Multiple Inputs Model
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.utils import plot_model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

visible1 = Input((64,64,3))
conv11 = Conv2D(32, kernel_size = 3, activation = 'relu')(visible1)
pool11 = MaxPooling2D(pool_size = (2,2))(conv11)
conv12 = Conv2D(16, kernel_size = 3, activation = 'relu')(pool11)
pool12 = MaxPooling2D(pool_size = (2,2))(conv12)
flat1 = Flatten()(pool12)

visible2 = Input((32,32,3))
conv21 = Conv2D(32, kernel_size = 3, activation = 'relu')(visible2)
pool21 = MaxPooling2D(pool_size = (2,2))(conv21)
conv22 = Conv2D(16, kernel_size = 3, activation = 'relu')(pool21)
pool22 = MaxPooling2D(pool_size = (2,2))(conv22)
flat2 = Flatten()(pool22)

merge = concatenate([flat1, flat2])

hidden1 = Dense(10, activation = 'relu')(merge)
hidden2 = Dense(10, activation = 'relu')(hidden1)
output = Dense(1, activation = 'sigmoid')(hidden2)

model = Model(inputs = [visible1, visible2], outputs = output)
model.summary()
plot_model(model, '7_Multiple_Input_Model.png')
