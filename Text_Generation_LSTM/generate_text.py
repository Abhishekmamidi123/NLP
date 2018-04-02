import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model
from keras.utils import np_utils

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

# map integers to characters
integer_to_char = []
for integer, char in enumerate(chars):
    integer_to_char.append((integer, char))
integer_to_char = dict(integer_to_char)
print integer_to_char

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
X = np.reshape(X_train, (samples, length_of_sequence, 1)) 
X = X/float(len(chars))
print X.shape

# y_train: Integers to one hot vectors
y_train = np_utils.to_categorical(y_train)
print y_train.shape

# Model
model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation = 'softmax'))
print model.summary()

# Load model weights
filename = 'model_weights_TG_5_epochs.hdf5'
model = load_model(filename)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

random_number = np.random.randint(0, len(X)-1)
input_sequence = X_train[random_number]
print input_sequence

# Prints the input_sequence in string format
input_sequence_char = []
for x in input_sequence:
    input_sequence_char.append(integer_to_char[x])
print ''.join(input_sequence_char)
print len(input_sequence)

output = []
output += input_sequence
for i in range(1000):
    x = np.reshape(input_sequence, (1, len(input_sequence), 1))
    x = x/float(len(chars))
    y_pred = model.predict(x, verbose=0)
    # print y_pred
    # print y_pred.shape
    index = np.argmax(y_pred)
    output.append(index)
    y_char = integer_to_char[index]
    # print index, y_char
    # print input_sequence
    input_sequence.append(index)
    input_sequence = input_sequence[1:]
    # print input_sequence
output = ''.join([integer_to_char[integer] for integer in output])
print output
