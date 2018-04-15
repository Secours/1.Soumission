'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle


from sys import argv, path
path.append ("../scoring_program")    # Contains libraries you will need
path.append ("../ingestion_program")  # Contains libraries you will need
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import  BaseEstimator

from prepro import Preprocessor

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC



class model:
    """Ce constructeur est supposé initialiser les membres de données.
         Utilisez des guillemets pour la documentation de la fonction.
         Le modèle est la classe appelée par Codalab.
         Cette classe doit avoir au moins une méthode "fit" et une méthode "predict"."""
    def __init__(self):
       
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        # Le modèle doit être défini dans le constructeur
        self.mod = Pipeline([
                ('preprocessing', Preprocessor()),#on fait appel au preprossessing avec le pipeline 
                ('predictor', Predictor())])
        print("MODEL=" + self.mod.__str__())

    def fit(self, X, y):
        """ Cette fonction devrait entraîner les paramètres du modèle.
         Ici, nous ne faisons rien dans cet exemple ...
         Args:
             X: Matrice de données d'apprentissage de dim num_train_samples * num_feat.
             y: Matrice d'étiquettes d'apprentissage de dim num_train_samples * num_labels.
         Les deux entrées sont des tableaux numpy.
         Pour la classification, les étiquettes peuvent être soit des nombres 0, 1, ... c-1 pour c classe
         ou un vecteur codé à chaud unique de zéros, avec un 1 à la kième position pour la classe k.
         Le format AutoML prend en charge le codage à chaud, qui fonctionne également pour les problèmes de multi-étiquettes"""
        
        self.num_train_samples = len(X)
        if X.ndim>1: self.num_feat = len(X[0])
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = len(y)
        if y.ndim>1: self.num_labels = len(y[0])
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
           # C'est ici que training a lieu
        
        self.mod.fit(X, y)
        self.is_trained=True

    def predict(self, X):
        """Cette fonction devrait fournir des prédictions d'étiquettes sur les données (de test).
         Ici nous venons de renvoyer des zéros ...
        """
        num_test_samples = len(X)
        if X.ndim>1: num_feat = len(X[0])
        print("PREDICT input: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
            # C'est ici que les prédictions sont faites.
         
        try:
            Y = self.mod.predict_proba(X)
            if Y.ndim>1: 
                num_labels = len(Y[0])
                print("PREDICT PROBA: number of labels=".format(num_labels))
                Y = Y[:,1]
                print("PREDICT PROBA: keeping only second column")
        except:
            Y = self.mod.predict(X)
        
        num_labels = 1
        if Y.ndim>1: num_labels = len(Y[0])
        print("PREDICT output: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        if (self.num_labels != num_labels):
            print("ARRGH: number of labels in X does not match training data!")
            
        return Y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
        



#classe pour definir le classifieur RandomForest
class Predictor(BaseEstimator):
    """Predictor: c'est dans cette classe qu'on cree RandomForestClassifier
    pour lequel on choisit les hyper-paramètres.  """
    def __init__(self):
        '''cette methode initialise le classifieur '''
        self.mod = RandomForestClassifier(n_estimators=100, max_features=100)
        print("PREDICTOR=" + self.mod.__str__())

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.mod = self.mod.fit(X, y)
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.mod.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
    
 
#classe pour definir le classifieur Gaussian  
class Gaussian(BaseEstimator):
    """Predictor: c'est dans cette classe qu'on cree RandomForestClassifier pour lequel on choisit les hyper-paramètres.  """
    def __init__(self):
        '''cette methode initialise le classifieur .'''
        self.mod=GaussianNB()
        print("Gaussian=" + self.mod.__str__())
    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.mod = self.mod.fit(X, y)
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.mod.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self


#classe pour definir le classifieur Kneighbors 
class Kneighbors (BaseEstimator):
    def __init__(self):
        '''This method initializes the predictor.'''
        self.mymodel=KNeighborsClassifier()
        
        print("Kneighbors=" + self.mymodel.__str__())

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.mymodel = self.mymodel.fit(X, y)
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.mymodel.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
    
#classe pour definir le classifieur ExtraTrees 
class ExtraTrees (BaseEstimator):
    def __init__(self):
        '''This method initializes the predictor.'''
        self.mymodel=ExtraTreesClassifier()
        print("ExtraTrees=" + self.mymodel.__str__())

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.mymodel = self.mymodel.fit(X, y)
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.mymodel.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
    
#classe pour definir le classifieur MLP
class MLP (BaseEstimator):
    def __init__(self):
        '''This method initializes the predictor.'''
        self.mymodel=MLPClassifier()
       
        print("MLP=" + self.mymodel.__str__())

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.mymodel = self.mymodel.fit(X, y)
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.mymodel.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
    
#classe pour definir le classifieur SVC   
class SVc (BaseEstimator):
    def __init__(self):
        '''This method initializes the predictor.'''
        self.mymodel=SVC()
        
        print("SVC=" + self.mymodel.__str__())

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.mymodel = self.mymodel.fit(X, y)
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.mymodel.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
    
    
#classe pour definir le classifieur GradientBoosting
class GradientBoosting (BaseEstimator):
    def __init__(self):
        '''This method initializes the predictor.'''
        self.mymodel=GradientBoostingClassifier()
        print("GradientBoosting=" + self.mymodel.__str__())

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.mymodel = self.mymodel.fit(X, y)
        return self

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.mymodel.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
      
        
if __name__=="__main__":
   
    if len(argv)==1: # Utiliser les répertoires d'entrée et de sortie par défaut si aucun argument n'est fourni
        input_dir = "../../public_data" # pour importer les donnees
        output_dir = "../results" 
        code_dir = "../starting_kit/ingestion_program" 
        metric_dir = "../starting_kit/scoring_program" 
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        code_dir = argv[3]
        metric_dir = argv[4]
        
    path.append (code_dir)
    path.append (metric_dir)
    
    metric_name = 'auc_metric_'
    import my_metric 
    scoring_function = getattr(my_metric, metric_name)
    print 'Using scoring metric:', metric_name
            
    from data_manager import DataManager    
    basename = 'Opioids'
    D = DataManager(basename, input_dir) 
    print D
    
   #Nous définissons ici sept modèles etle preprocessor et a chaque fois on 
   # compare un modele avec preprocessor grace au pipelinr
    model_dict = {
            'BasicPred': Predictor(),
            'Pipeline': Pipeline([('prepro', Preprocessor()), ('predictor', Predictor())]),
            'basicpredgb': GradientBoosting(),
            'PipelineGF': Pipeline([('prepro', Preprocessor()), ('predictorgb', GradientBoosting())]),
            'basicpredGNB': Gaussian(),
           
            'PipelineGNB': Pipeline([('prepro', Preprocessor()), ('Gaussian', Gaussian())]),
            'basicpredKN': Kneighbors(),
            'PipelineKN': Pipeline([('prepro', Preprocessor()), ('Kneighbors', Kneighbors())]),
            'basicpredET':  ExtraTrees(),
            'PipelineET': Pipeline([('prepro', Preprocessor()), ('ExtraTrees', ExtraTrees())]),

            'basicpredMLP': MLP(),
            'PipelineMLP': Pipeline([('prepro', Preprocessor()), ('MLP', MLP())]),
            'basicpredSVC':  SVc(),
            'PipelineSVC': Pipeline([('prepro', Preprocessor()), ('SVc', SVc())])
            }
     
    for key in model_dict:
        print("model....********")
        mymodel = model_dict[key]
        print("\n\n *** Model {:s}:{:s}".format(key,model_dict[key].__str__()))
 
        # Train
        print("Training")
        X_train = D.data['X_train']
        Y_train = D.data['Y_train']
        mymodel.fit(X_train, Y_train)
    
        # Predictions on training data
        print("Predicting")
        Ypred_tr = mymodel.predict(X_train)
        
        # Cross-validation predictions
        print("Cross-validating")
        from sklearn.model_selection import KFold
        from numpy import zeros  
        n = 3# 10-fold cross-validation
        kf = KFold(n_splits=n)
        kf.get_n_splits(X_train)
        Ypred_cv = zeros(Ypred_tr.shape)
        i=1
        for train_index, test_index in kf.split(X_train):
            print("Fold{:d}".format(i))
            Xtr, Xva = X_train[train_index], X_train[test_index]
            Ytr, Yva = Y_train[train_index], Y_train[test_index]
            mymodel.fit(Xtr, Ytr)
            Ypred_cv[test_index] = mymodel.predict(Xva)
            i = i+1
            

        # Compute and print performance
        training_score = scoring_function(Y_train, Ypred_tr)
        cv_score = scoring_function(Y_train, Ypred_cv)
        
        print("\nRESULTS FOR SCORE {:s}".format(metric_name))
        print("TRAINING SCORE= {:f}".format(training_score))
        print("CV SCORE= {:f}".format(cv_score))
        
    