# Sequential model
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential

model = Sequential()
model.add(Dense(2, input_dim = 1))
model.add(Dense(1))
