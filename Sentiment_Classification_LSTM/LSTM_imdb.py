import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

np.random.seed(7)
number_of_words = 5000
max_length_of_input = 500

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)
X_train = sequence.pad_sequences(X_train, max_length_of_input)
X_test = sequence.pad_sequences(X_test, max_length_of_input)

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(number_of_words, embedding_vector_length, input_length = max_length_of_input))
model.add(LSTM(100))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print model.summary()

model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))
score, accuracy = model.evaluate(X_test, y_test)
print accuracy
