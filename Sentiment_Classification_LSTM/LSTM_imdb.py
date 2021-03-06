import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import plot_model

np.random.seed(7)
number_of_words = 5000
max_length_of_input = 500

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)
X_train = sequence.pad_sequences(X_train, max_length_of_input)
X_test = sequence.pad_sequences(X_test, max_length_of_input)

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(number_of_words, embedding_vector_length, input_length = max_length_of_input))
# Uncomment the below line if you want to use Dropout
# model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print model.summary()
plot_model(model, to_file='model_LSTM.png', show_shapes=True, show_layer_names=True)

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
score_train, accuracy_train = model.evaluate(X_train, y_train)
print score_train, accuracy_train
score_test, accuracy_test = model.evaluate(X_test, y_test)
print score_test, accuracy_test

model.save('model_LSTM.h5')
