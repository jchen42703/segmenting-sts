
# coding: utf-8

# In[2]:


from keras.models import load_model
model_path = '...'


# In[ ]:


model = load_model(model_path)
import numpy as np 
ct_test = np.load('ct_test.npy')
y_test = np.load('y_patches_test.npy')

model.evaluate(ct_test, y_test)
predict = model.predict(ct_test)

np.save('prediction.npy', predict)

