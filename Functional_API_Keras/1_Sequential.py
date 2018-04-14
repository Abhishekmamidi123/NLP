# Sequential model - Example code
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.utils import plot_model

# Define X
# Define y
model = Sequential()
model.add(Dense(2, input_dim = 1))
model.add(Dense(1, activation='softmax'))
model.summary()
plot_model(model, '1_Sequential.png')
