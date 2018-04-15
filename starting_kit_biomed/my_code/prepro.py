#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
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
    
    # initialiser notre choix pour la méthode de preprocessing qui est dans
    # ce cas PCA qui prend en paramètre 100 features
    def __init__(self):
    
        self.transformer = PCA(n_components=100)
        print("PREPROCESSOR=" + self.transformer.__str__())


    # cette fonction qui permet d’entraîner le model après avoir été transformé
    # avec PCA( n_components=100) projection de nos 243 features sur un espace 
    # de 100 features.
    def fit(self, X, y=None):
        print("PREPRO FIT")
        return self.transformer.fit(X, y)


    # cette implémentation permet de convertir une collection de données texte
    # en une matrice de comptage
    def fit_transform(self, X, y=None):
        print("PREPRO FIT_TRANSFORM")
        return self.transformer.fit_transform(X)


    # transforme notre ancien X_train après toutes les modifications en 
    # un nouvel X_train
    def transform(self, X, y=None):
        print("PREPRO TRANSFORM")
        return self.transformer.transform(X)
    
    
if __name__=="__main__":
    
    from sys import argv, path
    path.append ("../ingestion_program")
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
 
    # Pré-traiter les données et les charger dans D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])

    
    # ceci prouve que le prétraitement a bien fonctionné
    print("*** Transformed data ***")
    print D
    
       
    X_train= D.data['X_train']
    print("Dim of X after transformation = " +  str(X_train.shape))
    
    
    
