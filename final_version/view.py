#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
View.py is the script to produce subjective views by machine learning methods
Classifier is the Class to receive machine learning type, build classification methods
                  to train the data and make predictions
Get view is the function receive features and date, use Classifier class to return views
'''

import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



class Classifier:
    '''
    Classifier is a class that receives machine learning type, can build machine learning model 
    fit the data and make prediction 
    
    Data Member:
    class_type -- the string that stands for machine learning type, involve 8 machine learning classification methods in total
        
    Member function:
    initialize_model -- Build specific classifier object according to input machine learning type
    fit_predict -- Fit the train data and make prediction
    '''
    
    def __init__(self,class_type):
        self.type = class_type
    
    def initialize_model(self):
        if self.type == 'Bayesian':
            clf = GaussianNB()

        elif self.type == 'Random Forest':
            clf = RandomForestClassifier(n_estimators=30,max_features = 'sqrt', \
                                         random_state = 10,min_samples_leaf=5)

        elif self.type == 'GB':
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,\
                                             max_depth=1, random_state=0)

        elif self.type == 'Adaboosting':
            clf = AdaBoostClassifier(n_estimators=100)

        elif self.type == 'Decision Tree':
            clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=7, \
                                     max_features=None, max_leaf_nodes=None, \
                                     min_samples_leaf=10, \
                                     min_samples_split=20, min_weight_fraction_leaf=0.0, \
                                     presort=False, random_state=10, splitter='best')

        elif self.type == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=5)

        elif self.type == 'Logistic':
            clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

        elif self.type == 'SVM':
            clf = svm.SVC(C=0.8, kernel='rbf', gamma=2, decision_function_shape='ovr')
            # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
        else:
            print('Attention!! No match type !! Try again !!')
        
        return clf

           
    def fit_predict(self,x_transformed,y):
        '''
        
        Arguments:
        x_transform -- x_transform is the standardized features. Because this is a rolling prediction problem. 
                       The train data is all information before date N, and the only test data is date N.
        y -- y is the label of train data for specific classification problem
        
        Returns:
        The prediction value for classification problem (About next week's return)
        '''        
        
        # Initialize machine learning model
        cl = self.initialize_model()
        # Divide data into train_x and test_x
        N = len(x_transformed)
        x_train = x_transformed[:N-1,:]
        y_train = y[:N-1]
        
        # Fit the model according to the machine learning model we choose
        cl.fit(x_train,y_train)
        
        # return the prediction value of the classification model
        return cl.predict(x_transformed[N-1,:].reshape(1,-1))[0]
    



def get_view(feature_list_Y1, feature_list_Y2, class_type, predict_date,Ndata):
    
    '''
    Get view receive feature list and predict date, use Classifier class to make prediction
    Then combined two classification problem together to produce subjective view on ETFs 
    (Y1 -- (1,-1), Y2 -- (1,2) --> Y (-2->very bearish, -1->bearish, 1->bullish, 2->very bullish)
    
    Arguments:
    feature_list_Y1 -- produced by feature_intro function, is the feature list for the first classification problem
    feature_list_Y2 -- produced by feature_intro function, is the feature list for the second classification problem
    class_type -- string type for machine learning method, ex: 'Random Forest', there are 8 choices in total
    predict_date -- the date that we want to make prediction 
    Ndata -- the window length for how many historical dates we want to use to train the model
    
    Returns:
    view -- the subjective view for each ETF [-2,-1,1,2]
    result -- the argument saved for backtest prediction accuracy
    '''
    
    # Initialization 
    view = np.zeros(len(feature_list_Y1))
    result = np.zeros([3,len(feature_list_Y1)])
    
    print(predict_date)
    for i in range(len(feature_list_Y1)):

        # Get the index of feature
        # ---------------- Y1 -------------------
        ix = np.where(feature_list_Y1[i].index == predict_date)[0][0]
        
        # number of features for this ticker
        n_feature = feature_list_Y1[i].shape[1]-1
        # Take the historical data accoriding to window length "Ndata"
        input_x = feature_list_Y1[i].iloc[ix-Ndata:ix,:n_feature]
        input_y = feature_list_Y1[i].iloc[ix-Ndata:ix,n_feature]
        
        # Feature scaling 
        pipeline = StandardScaler()
        input_x_transform = pipeline.fit_transform(input_x)
        
        # Use Classifier to build classification model
        clf = Classifier(class_type)
        # Make prediction
        y_pre1 =clf.fit_predict(input_x_transform,input_y)
        result[0,i] = (y_pre1 == input_y[-1]) # Check if the prediction is accurate or not
        # -------------- Y2 ------------------
        # Similarily, we can get prediction of Y2
        
        input_x2 = feature_list_Y2[i].iloc[ix-Ndata:ix,:n_feature]
        input_y2 = feature_list_Y2[i].iloc[ix-Ndata:ix,n_feature]
        
        # Feature scaling 
        pipeline = StandardScaler()
        input_x2_transform = pipeline.fit_transform(input_x2)
        
        # Use Classifier to build classification model and make prediction
        clf = Classifier(class_type)
        y_pre2 =clf.fit_predict(input_x2_transform,input_y2)
        result[1,i] = (y_pre2 == input_y2[-1])
        
        # ---------- Combine two predictions together -------------
        view[i] = y_pre1 * y_pre2
        result[2,i] = (view[i] == (input_y[-1]*input_y2[-1]))
        
    return [view,result]

