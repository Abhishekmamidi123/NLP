### Notes
- Use Stacked LSTM, if necessary to get good results instead of LSTM.
- CNN + LSTM - LSTM on top of CNN
  - CNN is used for feature extraction and LSTM is used to interpret the features accross the time steps.
- In Keras - Types of predictions:
  - model.predict(X)
  - model.predict_classes(X)
  - model.predict_proba(X)
- Life cycle for LSTM models in Keras
  - Define Network
    - Input must be three dimensional - samples, timesteps, features.
    - Convert 2D dataset into 3D dataset.
    - input_shape argument expects a tuple - (number of timesteps, number of features)
    - Some of the standard activation functions that can be used in the output layer:
      - Regression - 'linear'
      - Binary Classification(2 classes) - 'sigmoid'
      - Multiclass Classification(>2 classes) - 'softmax'
    - Example:
      - model = Sequential()
      - model.add(LSTM(5, input_shape=(2,1)))
      - model.add(Dense(1))
      - model.add(Activation('sigmoid'))
      
  - Compile Network
    - It transforms the simple sequence of layers that we defined into a highly efficient series of matrix transforms in a format intended to be executed on your GPU or CPU, depending on how Keras is configured.
    - Example:
      - model.compile(optimizer='adam', loss='mean_squared_error')
    - Some of the standard loss functions that can be used in the output layer:
      - Regression - 'mean_squared_error'
      - Binary Classification(2 classes) - 'binary_crossentropy' or cross entropy
      - Multiclass Classification(>2 classes) - 'categorical_crossentropy' or Multiclass Logarithmic Loss
    - Some of the optimization algorithms:
      - Stochastic Gradient Descent - 'sgd'
      - Adam - 'adam'
      - RMSprop - 'rmsprop'
  - Fit Network
    - 
  - Evaluate Network
  - Make Predictions
 
 
### References:
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Life cycle of LSTM models - keras](https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/)
