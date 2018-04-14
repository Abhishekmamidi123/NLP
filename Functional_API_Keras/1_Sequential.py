# Sequential model - Example code
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential

# Define X
# Define y
model = Sequential()
model.add(Dense(2, input_dim = 1))
model.add(Dense(1, activation='softmax'))

model.compile(loss = 'categorial_crossentropy', metrics = ['accuracy'])
model.run(X, y)
y_test = model.predict(X_test)
