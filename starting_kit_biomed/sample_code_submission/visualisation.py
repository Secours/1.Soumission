# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
author:DIALLO Mamadou Sanou
"""


# les librairies python
import seaborn as sns; sns.set()
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

#importation de la base de données data
model_dir = 'sample_code_submission/'          
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
from sys import path; path.append(model_dir); path.append(problem_dir); path.append(score_dir); 


datadir = 'sample_data'              # Change this to the directory where you put the input data
dataname = 'Opioids'
from data_io import read_as_df
data = read_as_df(datadir  + '/' + dataname) 
print(data.shape)

from data_manager import DataManager  
D = DataManager(dataname, datadir, replace_missing=True)

from model import model
M = model()
trained_model_name = model_dir + dataname
M = M.load(trained_model_name)                  # Attempts to re-load an already trained model



#Chaque graphe est codé dans une fonction qui lui est associéé

#Histogramme
def Histogramme():
    data[data.columns[:10]].hist(figsize=(10, 20), bins=50, layout=(5, 2))



#On commence par créer un tableau contenant les données les plus importantes, ici on fait abstraction du reste:

def extrait(tab,j):
    ligne,colonne=tab.shape
    temp = np.zeros(ligne)
    for i in range(ligne):
        temp[i]=tab[i][j]
    return temp 
    
# Construction d'un data frame reduit avec les variables importants    
    
x= np.zeros((5,data.shape[0]))
x[0]=extrait(data.values,0)
x[1]=extrait(data.values,1)
x[2]=extrait(data.values,2)
x[3]=extrait(data.values,3)
x[4]=extrait(data.values,243)
x=x.transpose()

y=np.zeros((17,data.shape[0]))

#On convertit ensuite le tableau obtenu en DataFrame pandas :

colonne=["Genger","State","Credentials","Speciality","Target"]
newData=pd.DataFrame(data=x,index=np.arange(data.shape[0]),columns=colonne)
#print(newData.head())

#Pairplot
 #Scatter plot : corrélation des données 
def Pairplot():
    sns.pairplot(newData[0:50],hue="Target")



#Heatmap de correaltion 
def HeatmapCorr():
    ax=sns.heatmap(data.corr())
    fig, ax = plt.subplots()
    fig.set_size_inches(48, 48)
    ax=sns.heatmap(data.corr())


#Construction des ensembles pour les graphs de performance
    
X_train = D.data['X_train']
Y_train = D.data['Y_train']
X_test=D.data['X_test']
if not(M.is_trained):    
    M.fit(X_train, Y_train)                     

Y_hat_train = M.predict(D.data['X_train']) # Optional, not really needed to test on taining examples
Y_hat_valid = M.predict(D.data['X_valid'])
Y_hat_test = M.predict(D.data['X_test'])


#CONFUSION MATRIX

#importation des metrics

from sklearn.metrics import confusion_matrix

Y_true = Y_train
X_test=(D.data['X_test'])
M.fit(X_train,Y_true)
Y_pred=M.predict(X_train)

print(Y_true.shape)
print(X_test.shape)

matrix_conf=confusion_matrix(Y_true,Y_pred)
#matrix_conf= matrix_conf/ matrix_conf.astype(np.float).sum(axis=1)    Normalisation de la matrice


# FOnction de tracage de la matrice de confusion


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

#print_confusion_matrix(matrix_conf,["bon medecin","mauvais medecin"],(10,7),14)




#Courbe ROC

def ROC():
    
    
    from sklearn.datasets import make_classification
    X, Y = make_classification(n_samples=10000, n_features=2, n_classes=2,
                               n_repeated=0, n_redundant=0)
    
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    y_pred = logreg.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    conf = confusion_matrix(y_test, y_pred)
    
    
    from sklearn.metrics import roc_curve
    
    fpr_cl = dict()
    tpr_cl = dict()
    
    y_pred = logreg.predict(X_test)
    y_proba = logreg.predict_proba(X_test)
    
    fpr_cl["classe 0"], tpr_cl["classe 0"], _ = roc_curve(
        y_test == 0, y_proba[:, 0].ravel())
    fpr_cl["classe 1"], tpr_cl["classe 1"], _ = roc_curve(
        y_test, y_proba[:, 1].ravel())  # y_test == 1
    
    import numpy
    prob_pred = numpy.array([y_proba[i, 1 if c else 0]
                             for i, c in enumerate(y_pred)])
    fpr_cl["tout"], tpr_cl["tout"], _ = roc_curve(
        (y_pred == y_test).ravel(), prob_pred)
    
    plt.figure()
    for key in fpr_cl:
        plt.plot(fpr_cl[key], tpr_cl[key], label=key)
    
    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Mauvais medecins incorrectement detectes")
    plt.ylabel("Bon medecins effectivement bien detectes")
    plt.title('ROC(s) avec Cross-Validation(proba)')
    plt.legend(loc="lower right")

#ROC()



# Test de toutes ces fonctions ecrites plus haut

if __name__=="__main__":
    print("Histogramme des variables:features")
    Histogramme()
    print("pairplot")
    Pairplot()
    print("Heatmap de Correlation")
    HeatmapCorr()
    print("Mtrice de Confusion")
    print_confusion_matrix(matrix_conf,["bon medecin","mauvais medecin"],(10,7),14)
    print("Courbe ROC")
    ROC()
