from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.utils import np_utils
from keras.utils import plot_model
from pickle import dump
from keras.utils import to_categorical

# Read sentences.
filename = 'rebuplic_preprocessed.txt'
file = open(filename, 'r')
data = file.read()
file.close()
lines = data.split('\n')

# Convert the words into integers.
t = Tokenizer()
t.fit_on_texts(lines)
encoded_lines = t.texts_to_sequences(lines)
vocab_size = len(t.word_index) + 1

# Split the data into X and y
encoded_lines = np.array(encoded_lines[:-1]) # Remove last line which is '\n'
X = encoded_lines[:,:-1]
y = encoded_lines[:,-1]
seq_length = len(X[1])

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length = seq_length))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print model.summary()

y = to_categorical(y, num_classes = vocab_size)
model.fit(X,y,batch_size=128, epochs=100)

model.save('word_generation_100_epochs.h5')
dump(t, open('tokenizer.pkl', 'wb'))
