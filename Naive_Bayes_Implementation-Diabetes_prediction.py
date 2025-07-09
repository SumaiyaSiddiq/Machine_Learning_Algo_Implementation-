# -*- coding: utf-8 -*-
"""

@author: sumaiya
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

df = pd.read_csv("diabetes_Dataset.csv")

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness',
                     'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values
y = df[predicted_class_names].values

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33)

print('\n the total number of Training Data :', ytrain.shape)
print('\n the total number of Test Data :', ytest.shape)

clf = GaussianNB().fit(xtrain, ytrain.ravel())
predicted = clf.predict(xtest)
predictTestData = clf.predict([[1, 189, 60, 23, 846, 30.1, 0.398, 59]])

print('\n Confusion matrix')
print(metrics.confusion_matrix(ytest, predicted))

print('Accuracy of the classifier is', metrics.accuracy_score(ytest, predicted))
print('The value of Precision', metrics.precision_score(ytest, predicted))
print('The value of Recall', metrics.recall_score(ytest, predicted))
print("Predicted Value for individual Test Data:", predictTestData)

