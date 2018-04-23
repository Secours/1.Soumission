# -*- coding: utf-8 -*-


# Histogrammes des algorithmes en fonction des resultats

 
import matplotlib.pyplot as plt
import numpy as np
 

fig = plt.figure()
x = np.arange(27)

height =np.array([0.987,0.6,0.65,0,0.99,0.52,0.55,0,0.65,0.59,0.62,0,0.62,0.51,0.54,0,0.89,0.18,0.22,0,0.65,0.49,0.53,0,0.176,0.17,0.21])
width = 1.0

 
gris=np.arange(1,27,4)
navy=np.arange(2,27,4)

graph=plt.bar(x, height, width, color='b')


for i in gris:

 
    graph[i].set_facecolor('gainsboro')

for j in navy:

 
    graph[j].set_facecolor('navy')

    

 
plt.bar(1, 0, 0.0, color='b', label='Training')
plt.bar(1, 0, 0.0, color='gainsboro', label='cross-validation')
plt.bar(1, 0, 0.0, color='navy', label='validation')


BarName = [' ','RandomForest ',' ',' ',' '  ,' DecisionTree',' ',' ',' ' ,'GradientBoosting',' ',' ','',' MLP' ,' ',' ','', 'SVM',' ' ,' ', '','KNeighbors',' ',' ' ,'', 'GaussianNB'] 
plt.xticks(x, BarName, rotation=90)
plt.savefig('Diagramme-des-algos1.png')

plt.title("Diagramme des resultats des differents algorithmes testes avec le prepro")

 
plt.legend()

#plt.xlim(0)

plt.show()



fig2 = plt.figure()
x2 = np.arange(27)

height =np.array([0.94,0.77,0.81,0,0.519,0.525,0.547,0,0.51,0.81,0.828,0,0.85,0.73,0.776,0,0.88,0.45,0.47,0,0.76,0.64,0.68,0,0.510,0.517,0.54])

graph2=plt.bar(x2, height, width, color='b')

for i in gris:
    graph2[i].set_facecolor('gainsboro')

for j in navy:
    graph2[j].set_facecolor('navy')

    
plt.bar(1, 0, 0.0, color='b', label='Training')
plt.bar(1, 0, 0.0, color='gainsboro', label='cross-validation')
plt.bar(1, 0, 0.0, color='navy', label='validation')



BarName = [' ','RandomForest ',' ',' ',' '  ,' DecisionTree',' ',' ',' ' ,'GradientBoosting',' ',' ','',' MLP' ,' ',' ','', 'SVM',' ' ,' ', '','KNeighbors',' ',' ' ,'', 'GaussianNB'] 
plt.xticks(x2, BarName, rotation=90)
plt.savefig('Diagramme-des-algos2.png')

plt.title("Diagramme des resultats des differents algorithmes testes sans le prepro")

 
plt.legend()

#plt.xlim(0)

plt.show()

fig3 = plt.figure()
x3 = np.arange(27)

height3 =np.array([0.198,0.41,0.60,0,0.17,0.349,0.51,0,0.194,0.402,0.593,0,0.054,0.122,0.182,0,0.063,0.124,0.178,0,0.163,0.339,0.497,0,0.169,0.35,0.52])

graph3=plt.bar(x3, height3, width, color='b')


for i in gris:
 
    graph3[i].set_facecolor('gainsboro')

for j in navy:

 
    graph3[j].set_facecolor('navy')
 
plt.bar(1, 0, 0.0, color='b', label='fold1')
plt.bar(1, 0, 0.0, color='gainsboro', label='fold2')
plt.bar(1, 0, 0.0, color='navy', label='fold3')


BarName3 = [' ','RandomForest ',' ',' ',' '  ,' MLP',' ',' ',' ' ,'GradientBoosting',' ',' ','',' SVM' ,' ',' ','', 'GAUSSIAN',' ' ,' ', '','KNeighbors',' ',' ' ,'', 'Extratrees'] 
plt.xticks(x3, BarName3, rotation=90)
plt.savefig('Diagramme-des-algos3.png')

plt.title("Resultat pour les differents folds avec le prepro")

# nb: one ne peut pas mettre les accents
plt.legend()

#plt.xlim(0)

plt.show()

fig4 = plt.figure()
x4 = np.arange(27)

height4 =np.array([0.21,0.425,0.626,0,0.195,0.383,0.57,0,0.2,0.419,0.626,0,0.064,0.139,0.2,0,0.11,0.21,0.31,0,0.161,0.334,0.488,0,0.204,0.418,0.616])
graph4=plt.bar(x4, height4, width, color='b')

for i in gris:

 
  
    graph4[i].set_facecolor('gainsboro')

for j in navy:

 
    graph4[j].set_facecolor('navy')

 
plt.bar(1, 0, 0.0, 
color='b', 
label='fold1')
plt.bar(1, 0, 0.0, color='gainsboro', label='fold2')
plt.bar(1, 0, 0.0, color='navy', label='fold3')


BarName = [' ','RandomForest ',' ',' ',' '  ,' MLP',' ',' ',' ' ,'GradientBoosting',' ',' ','',' SVM' ,' ',' ','', 'GAUSSIAN',' ' ,' ', '','KNeighbors',' ',' ' ,'', 'Extratrees'] 
plt.xticks(x4, BarName, rotation=90)
plt.savefig('Diagramme-des-algos4.png')

plt.title("Resultat pour les differents folds sans le prepro")
# nb: one ne peut pas mettre les accents
plt.legend()

#plt.xlim(0)

plt.show()
