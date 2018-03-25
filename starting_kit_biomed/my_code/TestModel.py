#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: isabelleguyon

This is an example of predictive model program, we show how to combine
a predictive model and a preprocessor with a pipeline. 

IMPORTANT: keep calling your program model.py so the ingestion program that
runs on Codalab can find it.
"""

from sys import argv, path
import numpy as np
import pickle
from os.path import isfile
path.append ("../scoring_program")    # Contains libraries you will need
path.append ("../ingestion_program")  # Contains libraries you will need


from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        #self.mymodel=GaussianNB()
        self.mymodel=RandomForestClassifier(n_estimators=100, max_features=100)
        #self.mymodel=KneighborsClassifier()
        #self.mymodel=ExtraTreesClassifier()
        #self.mymodel=MLPClassifier()
        #self.mymodel=SVC()
        #self.mymodel=GradientBoostingClassifier()
        

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        self.num_train_samples = len(X)
        if X.ndim>1: self.num_feat = len(X[0])
        print("FIT: dim(X)= [{:d}, {:d}]").format(self.num_train_samples, self.num_feat)
        num_train_samples = len(y)
        if y.ndim>1: self.num_labels = len(y[0])
        print("FIT: dim(y)= [{:d}, {:d}]").format(num_train_samples, self.num_labels)
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        else:
            self.mymodel = self.mymodel.fit(X, y)
            self.is_trained=True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = len(X)
        if X.ndim>1: num_feat = len(X[0])
        print("PREDICT: dim(X)= [{:d}, {:d}]").format(num_test_samples, num_feat)
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]").format(num_test_samples, self.num_labels)
        y = self.mymodel.predict_proba(X)[:,1]
        return y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self



if __name__=="__main__":
    # We can use this to run this file as a script and test the Classifier
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../sample_data" # A remplacer par public_data
        output_dir = "../results"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
                         
    from sklearn.metrics import accuracy_score      
    # Interesting point: the M2 prepared challenges using sometimes AutoML challenge metrics
    # not scikit-learn metrics. For example:
    from libscores import auc_metric
                 
    from data_manager import DataManager 
    from data_converter import convert_to_num 
    
    basename = 'Opioids'
    D = DataManager(basename, input_dir) # Load data
    print D
    
    # Here we define 3 classifiers and compare them
    classifier_dict = {
            
            'RandomPred': RandomForestClassifier(),
            'BasicPred': KNeighborsClassifier(),
            'FancyPred': GaussianNB()}
        
    
    print "Classifier\tAUC\tACC"
    for key in classifier_dict:
        myclassifier = classifier_dict[key]
 
        # Train
        Yonehot_tr = D.data['Y_train']
        # Attention pour les utilisateurs de problemes multiclasse,
        # mettre convert_to_num DANS la methode fit car l'ingestion program
        # fournit Yonehot_tr a la methode "fit"
        # Ceux qui resolvent des problemes a 2 classes ou des problemes de
        # regression n'en ont pas besoin
        Ytrue_tr = convert_to_num(Yonehot_tr, verbose=False) # For multi-class only, to be compatible with scikit-learn
        myclassifier.fit(D.data['X_train'], Ytrue_tr)
        
        # Some classifiers and cost function use a different encoding of the target
        # values called on-hot encoding, i.e. a matrix (nsample, nclass) with one at
        # the position of the class in each line (also called position code):
        #nclass = len(set(Ytrue_tr))
        #Yonehot_tr = np.zeros([Ytrue_tr.shape[0],nclass])
        #for i, item in enumerate(Ytrue_tr): Yonehot_tr[i,item]=1
    
        # Making classification predictions (the output is a vector of class IDs)
        Ypred_tr = myclassifier.predict(D.data['X_train'])
        Ypred_va = myclassifier.predict(D.data['X_valid'])
        Ypred_te = myclassifier.predict(D.data['X_test'])  
        
        # Making probabilistic predictions (each line contains the proba of belonging in each class)
        Yprob_tr = myclassifier.predict_proba(D.data['X_train'])
        Yprob_va = myclassifier.predict_proba(D.data['X_valid'])
        Yprob_te = myclassifier.predict_proba(D.data['X_test']) 
    
        # Training success rate and error bar:
        # First the regular accuracy (fraction of correct classifications)
        acc = accuracy_score(Ytrue_tr, Ypred_tr)
        # Then two AutoML challenge metrics, working on the other representation
        auc = auc_metric(Yonehot_tr, Yprob_tr, task='multiclass.classification')
        #bac = bac_metric(Yonehot_tr, Yprob_tr, task='multiclass.classification')
        # Note that the AutoML metrics are rescaled between 0 and 1.
        
        print "%s\t%5.2f\.2f\t%5.2f\)" % (key, auc, acc)
        print "The error bar is valid for Acc only"
        # Note: we do not know Ytrue_va and Ytrue_te
        # See modelTest for a better evaluation using cross-validation
        
    # Another useful tool is the confusion matrix
    from sklearn.metrics import confusion_matrix
    print "Confusion matrix for %s" % key
    print confusion_matrix(Ytrue_tr, Ypred_tr)
    # On peut aussi la visualiser, voir:
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    # Voir aussi http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
