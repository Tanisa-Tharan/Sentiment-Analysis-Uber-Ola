"""
Title: Movie Review Sentiment Analysis
Author: Abhinav Thukral
Description: Implemented text analysis using machine learning models to classify movie review sentiments as positive or negative.
"""

#Importing Essentials
import pandas as pd
from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC 

#Plotly imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#Basic imports

import numpy as np
from tqdm import tqdm
import time

#Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

#General Sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

#Hyperopt imports
import hyperopt as hp
from hyperopt import fmin, tpe, Trials, hp, STATUS_OK, space_eval
from hyperopt.pyll.base import scope


#Classification imports
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

#Ignore warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


path = 'data/Uber_Dataset1.csv'
data = pd.read_csv(path)
data=data[data.Analysis!="Neutral"]
X = data.tweet
y = data.label
#Using CountVectorizer to convert text into tokens/features
vect = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=101, test_size= 0.2)
#Using training data to transform text into counts of features for each message
vect.fit(X_train)
X_train_dtm = vect.transform(X_train) 
X_test_dtm = vect.transform(X_test)

#Accuracy using Naive Bayes Model
NB = MultinomialNB()
NB.fit(X_train_dtm, y_train)
y_pred = NB.predict(X_test_dtm)
print('\nNaive Bayes')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')





from sklearn.model_selection import KFold
kf = KFold(n_splits=5)


x=vect.transform(X)   



# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
nb = MultinomialNB()
scores = cross_val_score(nb, x, y, cv=10, scoring='accuracy')
print(scores)


# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())





#Accuracy using SVM Model
SVM = LinearSVC()
SVM.fit(X_train_dtm, y_train)
y_pred = SVM.predict(X_test_dtm)
print('\nSupport Vector Machine')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')






def bayes_tuning(estimator, xdata, ydata, cv, space, max_it):
    
    #Define objective function
    def obj_function(params):
        model = estimator(**params)
        score = cross_val_score(estimator = model, X = xdata, y = ydata,
                                scoring = 'accuracy',
                                cv = cv).mean()
        return {'loss': -score, 'status': STATUS_OK}
    
    start = time.time()
    
    #Perform tuning
    hist = Trials()
    param = fmin(fn = obj_function, 
                 space = space,
                 algo = tpe.suggest,
                 max_evals = max_it,
                 trials = hist,
                 rstate = np.random.RandomState(1))
    param = space_eval(space, param)
    
    #Compute best score
    score = -obj_function(param)['loss']
    
    return param, score, hist, time.time() - start

sv_params = {'C': hp.uniform('C', 0.1, 2.0),
             'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
             'degree': scope.int(hp.quniform('degree', 2, 5, 1)),
             'gamma': hp.choice('gamma', ['auto', 'scale']),
             'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-2)),
             'max_iter': scope.int(hp.quniform('max_iter', -1, 100, 1))
}


models = [SVC]
model_params = [sv_params]
model_names=["SVC"]
bayes_score, bayes_time, bayes_hist = [], [], []
for m, par in tqdm(zip(models, model_params)):
    param, score, hist, dt = bayes_tuning(m, X_train_dtm, y_train, 10, par, 150)
    bayes_score.append(score)
    bayes_time.append(dt)
    bayes_hist.append(hist)
bayes_df = pd.DataFrame(index = model_names)
bayes_df['Accuracy'] = bayes_score
bayes_df['Time'] = bayes_time

print("Accuracy Rate of Bayseian Search CV (SVM) : "+str(bayes_df))


# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train_dtm, y_train) 

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 

grid_predictions = grid.predict(X_test_dtm) 
  
grid_score=metrics.accuracy_score(y_test,grid_predictions)
# print classification report 
print(classification_report(y_test, grid_predictions)) 

print('Accuracy Rate of GridSearch CV (SVM) :',metrics.accuracy_score(y_test,grid_predictions)*100,'%',sep='')









#Define function for random search tuning
def random_tuning(estimator, xdata, ydata, cv, space, max_iter):
    
    start = time.time()
    
    #Perform tuning
    rand = RandomizedSearchCV(estimator = estimator,
                              param_distributions = space,
                              n_iter = max_iter,
                              scoring = 'accuracy',
                              cv = 10,
                              random_state = np.random.RandomState(1))
    rand.fit(xdata, ydata)
    
    return rand.best_params_, rand.best_score_, rand.cv_results_['mean_test_score'], time.time() - start


sv_params = {'C': list(np.linspace(0.1, 2.0, 10)),
             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
             'degree': list(range(2, 6)),
             'gamma': ['auto', 'scale'],
             'tol': list(np.logspace(np.log(1e-5), np.log(1e-2), num = 10, base = 3)),
             'max_iter': list(range(-1, 101))
}




#Apply random seach tuning
models = [ SVC()]
model_params = [ sv_params]

rand_score, rand_time, rand_hist = [], [], []
for m, par in tqdm(zip(models, model_params)):
    _, score, hist, dt = random_tuning(m,X_train_dtm, y_train, 10, par, 150)
    rand_score.append(score)
    rand_time.append(dt)
    rand_hist.append(hist)
    
    
#Print random search tuning results
rand_df = pd.DataFrame(index = model_names)
rand_df['Accuracy'] = rand_score
rand_df['Time'] = rand_time

print("Accuracy Rate of RandomSearch CV (SVM) :"+str(rand_df))
















#Naive Bayes Analysis
tokens_words = vect.get_feature_names()
print('\nAnalysis')
print('No. of tokens: ',len(tokens_words))
counts = NB.feature_count_
df_table = {'Token':tokens_words,'Negative': counts[0,:],'Positive': counts[1,:]}
tokens = pd.DataFrame(df_table, columns= ['Token','Positive','Negative'])
positives = len(tokens[tokens['Positive']>tokens['Negative']])
print('No. of positive tokens: ',positives)
print('No. of negative tokens: ',len(tokens_words)-positives)
#Check positivity/negativity of specific tokens
#token_search = ['awesome']
#print('\nSearch Results for token/s:',token_search)
#print(tokens.loc[tokens['Token'].isin(token_search)])
#Analyse False Negatives (Actual: 1; Predicted: 0)(Predicted negative review for a positive review) 
#print(X_test[ y_pred < y_test ])
#Analyse False Positives (Actual: 0; Predicted: 1)(Predicted positive review for a negative review) 
#print(X_test[ y_pred > y_test ])

#Custom Test: Test a review on the best performing model (Logistic Regression)
#trainingVector = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 5)
#trainingVector.fit(X)
#X_dtm = trainingVector.transform(X)
#LR_complete = LogisticRegression()
#LR_complete.fit(X_dtm, y)
#Input Review
#print('\nTest a custom review message')
#print('Enter review to be analysed: ', end=" ")
#test = []
#test.append(input())
#test_dtm = trainingVector.transform(test)
#predLabel = LR_complete.predict(test_dtm)
#tags = ['Negative','Positive']
#Display Output
#print('The review is predicted',tags[predLabel[0]])


