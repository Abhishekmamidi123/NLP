import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import np_utils
from keras.utils import plot_model

# Read file - wonderland.txt
filename = 'wonderland.txt'
data = open(filename, 'r')
data = data.read().lower()
print "Length of the data: " + str(len(data))

# Find unique characters in the data
chars = sorted(list(set(data)))
print "Number of unique charaters in the data: " + str(len(chars))

# Map characters to integers
char_to_integer = []
for integer, char in enumerate(chars):
    char_to_integer.append((char, integer))
char_to_integer = dict(char_to_integer)
print char_to_integer

# Convert X(sequence) and y(one character) into integers
X_train = []
y_train = []
length_of_sequence = 100
for i in range(0, (len(data) - length_of_sequence)):
    sequence = data[i:i+length_of_sequence]
    sequence_int = []
    for char in sequence:
        sequence_int.append(char_to_integer[char])
    X_train.append(sequence_int)    
    label = data[i+length_of_sequence]
    y_train.append(char_to_integer[label])

# Reshape X_train and normalize
samples = len(X_train)
X_train = np.reshape(X_train, (samples, length_of_sequence, 1)) 
X_train = X_train/float(len(chars))
print X_train.shape

# y_train: Integers to one hot vectors
y_train = np_utils.to_categorical(y_train)
print y_train.shape

# Model
model = Sequential()
model.add(LSTM(256, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation = 'softmax'))
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 10, batch_size = 32)
model.save('model_weights_LSTM.h5')

# plot_model(model, to_file='model_LSTM.png', show_shapes=True, show_layer_names=True)
