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

print t.word_index
print encoded_data

vocab_size = len(t.word_index) + 1
print vocab_size

sequences = []
for i in range(1, len(encoded_data)):
    sequences.append(encoded_data[i-1:i+1])
sequences = np.array(sequences)
print sequences
X = sequences[:,0]
y = sequences[:,1]
print y
y = to_categorical(y, vocab_size)
print X
print y

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length = 1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation = 'softmax'))
print model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X, y, epochs = 100, verbose = 2)

