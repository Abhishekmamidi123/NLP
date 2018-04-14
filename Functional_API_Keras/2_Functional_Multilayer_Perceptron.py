# Functional API - Multilayer Perceptron
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.utils import plot_model

visible = Input(shape=(10,))
hidden1 = Dense(10, activation = 'relu')(visible)
hidden2 = Dense(20, activation = 'relu')(hidden1)
hidden3 = Dense(10, activation = 'relu')(hidden2)
output = Dense(1, activation = 'sigmoid')(hidden3)

model = Model(inputs = visible, outputs = output)
print model.summary()
plot_model(model, to_file = '2_Functional_Multilayer_perceptron.png')
