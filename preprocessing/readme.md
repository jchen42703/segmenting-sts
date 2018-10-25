# Preprocessing Procedures
* based on utility functions from `preprocess.py` 
* generates preprocessed patches for both training and label data (defaults to 56 x 56 patches)
* __Training/Test Images__
  * normalizes pixel intensities between [0,1] 
  * balances the dataset (by default)
* __Labels__
  * one hot encodes them

## Future Work
* include whitening and clipping (i.e. between the range [-5,5]
* include the preprocessing compatibility for PET scans as well
