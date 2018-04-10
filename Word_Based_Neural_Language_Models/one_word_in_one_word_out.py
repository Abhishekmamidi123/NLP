import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential

# Read text
data = """ Jack and Jill went up the hill\n
        To fetch a pail of water\n
        Jack fell down and broke his crown\n
        And Jill came tumbling after\n """

data = [data]
t = Tokenizer()
t.fit_on_texts(data)
encoded_data = t.texts_to_sequences(data)[0]

vocab_size = len(t.word_index) + 1
print vocab_size

sequences = []
for i in range(1, len(encoded_data)):
    sequences.append(encoded_data[i-1:i+1])
sequences = np.array(sequences)

# X and y
X = sequences[:,0]
y = sequences[:,1]
y = to_categorical(y, vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length = 1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation = 'softmax'))
print model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X, y, epochs = 1000, verbose = 0)

# Generate text
start_text = 'jack'
start_text = start_text.lower()

predicted_words = []
predicted_words.append(start_text)

N = 30
for i in range(N):
    input_sequence = t.texts_to_sequences([start_text])[0]
    y_pred = model.predict_classes(np.array(input_sequence))
    word_pred = ''
    for word,index in t.word_index.items():
        if index == y_pred:
            word_pred = word
            break
    predicted_words.append(word_pred)
    start_text = word_pred

output = ' '.join(predicted_words)
print output
