import numpy as np
from pickle import load
from random import randint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Read file
filename = 'rebuplic_preprocessed.txt'
file = open(filename, 'r')
data = file.read()
file.close()
lines = data.split('\n')
lines = lines[:-1]  # Remove last line - '\n'
seq_length = len(lines[0].split())-1

model = load_model('word_generation_100_epochs.h5')
t = load(open('tokenizer.pkl', 'rb'))

# Generate random text
random_number = randint(0, len(lines))
start_text = lines[random_number]

predicted_words = []
predicted_words.append(start_text)

N = 50
for i in range(N):
    encoded_input_sequence = t.texts_to_sequences([start_text])
    input_sequence = pad_sequences(encoded_input_sequence, seq_length, truncating='pre')
    y_pred = model.predict_classes(np.array(input_sequence))
    word_pred = ''
    for word,index in t.word_index.items():
        if index == y_pred:
            word_pred = word
            break
    predicted_words.append(word_pred)
    start_text = start_text + ' ' + word_pred

output = ' '.join(predicted_words)
print output
