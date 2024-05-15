# Importing the required libraries
import numpy as np
import os
from data import getData
from sklearn import svm
from sklearn.metrics import accuracy_score
from warnings import simplefilter
from sklearn.model_selection import GridSearchCV

# ignore all warnings
simplefilter(action='ignore')

X, y = getData()


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # LogisticRegression


from sklearn.linear_model import LogisticRegression  
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print(f'LogisticRegression Accuracy: {round(accuracy*100,3)}%')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # SVM
from sklearn.svm import SVC

clf = svm.SVC()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print(f'SVC Accuracy: {round(accuracy*100,3)}%')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # LGBM
from lightgbm import LGBMClassifier
 
lgbm = LGBMClassifier(verbose=-1)

lgbm.fit(X_train, y_train)

y_pred = lgbm.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print(f'LGBMClassifier Accuracy: {round(accuracy*100,3)}%')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Naive Bayes: Gaussian

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print(f'GaussianNB Accuracy: {round(accuracy*100,3)}%')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Naive Bayes: Multinomial


from sklearn.naive_bayes import MultinomialNB

# Initialize the Gaussian Naive Bayes classifier
nb = MultinomialNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print(f'MultinomialNB Accuracy: {round(accuracy*100,3)}%')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Stochastic Gradient Descent Classifier

from sklearn import linear_model

SGDClf = linear_model.SGDClassifier()
SGDClf.fit(X_train, y_train)

y_pred = SGDClf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print(f'SGDClassifier Accuracy: {round(accuracy*100,3)}%')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # KNeighbors 


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print(f'KNeighborsClassifier Accuracy: {round(accuracy*100,3)}%')
from sklearn.metrics import confusion_matrix
print (confusion_matrix(y_test, y_pred))
# KNeighborsClassifier Accuracy: 98.865%
# [[1280    4    9    0]
#  [   0 1286    6    0]
#  [   4   35 1259    0]
#  [   0    0    0 1229]]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Decision Trees 
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print(f'DecisionTreeClassifier Accuracy: {round(accuracy*100,3)}%')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Random Forest 

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print(f'RandomForestClassifier Accuracy: {round(accuracy*100,3)}%')