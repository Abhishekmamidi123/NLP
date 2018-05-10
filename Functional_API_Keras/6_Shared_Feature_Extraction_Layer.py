# Shared feature Extraction Layer model
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.utils import plot_model
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate

visible = Input((100,1))
extract1 = LSTM(10)(visible)

fe_layer_11 = Dense(10, activation = 'relu')(extract1)

fe_layer_21 = Dense(10, activation = 'relu')(extract1)
fe_layer_22 = Dense(20, activation = 'relu')(fe_layer_21)
fe_layer_23 = Dense(10, activation = 'relu')(fe_layer_22)

merge = concatenate([fe_layer_11, fe_layer_23])

output = Dense(1, activation = 'sigmoid')(merge)
model = Model(inputs = visible, outputs = output)

print model.summary()
plot_model(model, '6_Shared_Extraction_Layer.png', show_shapes=True, show_layer_names=True)
