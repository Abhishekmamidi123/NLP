import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# from keras.utils import np_utils

# Read file - wonderland.txt
filename = 'wonderland.txt'
data = open(filename, 'r')
data = data.read().lower()
print "Length of the data: " + str(len(data))

# Find unique characters in the data
chars = sorted(list(set(data)))
print "Number of unique charaters in the data: " + str(len(chars))

# Map characters to integers
chars_to_integers = []
for char, integer in enumerate(chars):
    chars_to_integers.append((char, integer))
print dict(chars_to_integers)


