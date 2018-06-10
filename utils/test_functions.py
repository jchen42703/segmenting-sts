
# coding: utf-8

# In[1]:


import numpy as np 
import os
import h5py


# In[3]:


def np_listcheck(list1, list2):
    """Compares two lists of numpy arrays and sees if the contents are the same.
    Args:
        two lists of numpy arrays
    Returns: 
        Boolean depending on if the contents are the same.
    """
    counter = 0
    
    for d,e in zip(list1, list2):  
        if np.array_equal(d,e):
                pass
        else:
                counter += 1
                print(counter)
                
    if counter == len(list1):
        print(False)
    else: 
        print(True)


# In[ ]:




