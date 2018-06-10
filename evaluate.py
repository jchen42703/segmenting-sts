from keras.models import load_model
model_path = '...'

model = load_model(model_path)

#load data
import numpy as np 
ct_test = np.load('ct_test.npy')
y_test = np.load('y_patches_test.npy')

#evaluation and prediction
model.evaluate(ct_test, y_test)
predict = model.predict(ct_test)

np.save('prediction.npy', predict)

