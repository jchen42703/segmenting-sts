import preprocess as pch
import numpy as np
import tensorflow as tf

np.random.seed(pch.SEED)
data_dir = 'data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    
#getting normalized numpy arrays for TRAINING DATA
ct_slices = pch.normalize(pch.get_slices(pch.ct_data, patient_ids = pch.patients['train'], use_pos_window = True))
pet_slices = pch.normalize(pch.get_slices(pch.pet_data, patient_ids = pch.patients['train'], use_pos_window = True))

test_ct_slices = pch.normalize(pch.get_slices(pch.ct_data, patient_ids = pch.patients['test'], use_pos_window = True))
test_pet_slices = pch.normalize(pch.get_slices(pch.pet_data, patient_ids = pch.patients['test'], use_pos_window = True))

pch.gen_patches(ct_slices, pet_slices, save = True)
pch.gen_patches(test_ct_slices, test_pet_slices, save = True)
pch.get_patch_labels(pch.y_data, save = True)
