# -*- coding: utf-8 -*-
"""
Created on Mon Apr 02 16:42:06 2018

@author: Timgad
"""


# Histogrammes des algorithmes en fonction des resultats

 
import matplotlib.pyplot as plt
import numpy as np
 

fig = plt.figure()
x = np.arange(27)

height =np.array([0.94,0.77,0.81024516,0,0.519,0.525,0.547353688,0,0.51,0.81,0.82874158,0,0.85,0.73,0.7760517,0,0.88,0.45,0.47623974,0,0.76,0.64,0.68656832,0,0.510,0.517,0.54722029])
width = 1.0

 
rouge=np.arange(1,27,4)
vert=np.arange(2,27,4)

graph=plt.bar(x, height, width, color='b')

 

for i in rouge:

 
    graph[i].set_facecolor('red')

for j in vert:

 
    graph[j].set_facecolor('green')

    

 
plt.bar(1, 0, 0.0, 
color='red', 
label='cross-validation')
plt.bar(1, 0, 0.0, color='b', label='Training')
plt.bar(1, 0, 0.0, color='green', label='validation')


BarName = [' ','RandomForest ',' ',' ',' '  ,' DecisionTree',' ',' ',' ' ,'GradientBoosting',' ',' ','',' MLP' ,' ',' ','', 'SVM',' ' ,' ', '','KNeighbors',' ',' ' ,'', 'GaussianNB'] 
plt.xticks(x, BarName, rotation=90)
plt.savefig('Diagramme-des-algos.png')

plt.title("Diagramme des resultats des differents algorithmes testes")

 

# nb: one ne peut pas mettre les accents

 

plt.legend()

#plt.xlim(0)

plt.show()