#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: isabelleguyon

This is an example of program that preprocesses data.
It calls the PCA function from scikit-learn.
Replace it with programs that:
    normalize data (for instance subtract the mean and divide by the standard deviation of each column)
    construct features (for instance add new columns with products of pairs of features)
    select features (see many methods in scikit-learn)
    perform other types of dimensionality reductions than PCA
    remove outliers (examples far from the median or the mean; can only be done in training data)
"""

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA



class Preprocessor(BaseEstimator):
    def __init__(self):
    
        self.transformer = PCA(n_components=100)
        print("PREPROCESSOR=" + self.transformer.__str__())

    def fit(self, X, y=None):
        print("PREPRO FIT")
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        print("PREPRO FIT_TRANSFORM")
        return self.transformer.fit_transform(X)

    def transform(self, X, y=None):
        print("PREPRO TRANSFORM")
        return self.transformer.transform(X)
    
    
if __name__=="__main__":
    # Put here your OWN test code
    
    # To make sure this runs on Codalab, put here things that will not be executed on Codalab
    from sys import argv, path
    path.append ("../starting_kit/ingestion_program")
    from data_manager import DataManager  # such as DataManager
    
    if len(argv)==1: 
        input_dir = "../public_data" 
        output_dir = "../results" 
    else:
        input_dir = argv[1]
        output_dir = argv[2];
    
    basename = 'Opioids'
    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print D
    
    Prepro = Preprocessor()
 
    # Preprocess on the data and load it back into D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])

    
    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print D
    
       
    X_train= D.data['X_train']
    print("Dim of X after transformation = " +  str(X_train.shape))
    
    
