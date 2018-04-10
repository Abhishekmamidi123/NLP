import numpy as np
from pickle import load
from random import randint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import load_model

# Read file
filename = 'rebuplic_preprocessed.txt'
file = open(filename, 'r')
data = file.read()
file.close()
lines = data.split('\n')
lines = lines[:-1]  # Remove last line - '\n'
seq_length = len(lines[0].split())-1

model = load_model('word_generation_100_epochs')
