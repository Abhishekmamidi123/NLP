# Multiple Output Model
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.utils import plot_model
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
from keras.layers.wrappers import TimeDistributed

visible = Input(shape = (100, 1))

hidden = LSTM(10, return_sequences = True)(visible)

class11 = LSTM(10)(hidden)
class12 = Dense(10, activation = 'relu')(class11)
output1 = Dense(1, activation = 'sigmoid')(class12)
output2 = TimeDistributed(Dense(1, activation = 'linear'))(hidden)

model = Model(inputs = visible, outputs = [output1, output2])
print model.summary()
plot_model(model, '8_Multiple_Output_Model.png', show_shapes=True, show_layer_names=True)
