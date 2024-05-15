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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # LogisticRegression

# from sklearn.linear_model import LogisticRegression

# # Define hyperparameters grid
# # param_grid = {
# #     'C': [0.001, 0.01, 0.1, 1, 10, 100],
# #     'penalty': ['l1', 'l2', 'elasticnet'],
# #     'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
# #     'multi_class': ['auto', 'ovr', 'multinomial'],
# #     'max_iter': [10, 50, 100, 200, 300, 400, 500]
# # }
# # Best parameters found:  {'C': 100.0, 'solver': 'sag'}
# # Best cross-validation score: 0.91
# param_grid = {
#     'C' : [0.01, 0.1, 1.0, 10.0, 100.0 ],
#     'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
# }
# # Create logistic regression model
# logistic = LogisticRegression()

# # Grid search cross-validation
# grid_search = GridSearchCV(estimator=logistic, param_grid=param_grid)
# grid_search.fit(X, y)

# # Print best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# # from sklearn.linear_model import LogisticRegression  
# # classifier = LogisticRegression()
# # classifier.fit(X_train, y_train)

# # y_pred = classifier.predict(X_test)

# # accuracy = accuracy_score(y_test,y_pred)

# # print(f'LogisticRegression Accuracy: {round(accuracy*100,3)}%')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # SVM
# from sklearn.svm import SVC
# # param_grid = {
# #     'C': [0.1, 1, 10, 100],
# #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
# #     'gamma': ['scale', 'auto', 0.1, 1, 10],
# #     'degree': [2, 3, 4],
# #     'coef0': [0.0, 0.1, 1.0]
# # }
# # Best parameters found:  {'C': 100, 'kernel': 'poly'}
# # Best cross-validation score: 0.92
# param_grid = {
#     'kernel': ['linear', 'poly', 'rbf'],  # Different kernel types
#     'C': [0.1, 1, 10, 100]  # Regularization parameter values
# }
# clf = svm.SVC()

# # Grid search cross-validation
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid)
# grid_search.fit(X, y)

# # Print best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # LGBM
# from lightgbm import LGBMClassifier
# # param_grid = {
# #     'learning_rate': [0.01, 0.1],
# #     'n_estimators': [100, 300],
# #     'max_depth': [3, -1],  # -1 means no limit
# #     'min_child_samples': [5, 20],
# #     'subsample': [0.6, 1.0],
# #     'colsample_bytree': [0.6, 1.0],
# #     'reg_alpha': [0, 1.0],
# #     'reg_lambda': [0, 1.0]
# # }
# param_grid = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Learning rate values
#     'n_estimators': [50, 100, 200, 300]  # Number of trees values
# }
# # Best parameters found:  {'learning_rate': 0.05, 'n_estimators': 100}
# # Best cross-validation score: 0.92
# lgbm = LGBMClassifier(verbose=-1)

# # Grid search cross-validation
# grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid)
# grid_search.fit(X, y)

# # Print best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Naive Bayes: Gaussian

# from sklearn.naive_bayes import GaussianNB
# # Best parameters found:  {'var_smoothing': 1e-09}
# # Best cross-validation score: 0.68
# param_grid = {
#     'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # Variation smoothing parameter
# }
# # Initialize the Gaussian Naive Bayes classifier
# gnb = GaussianNB()

# # Grid search cross-validation
# grid_search = GridSearchCV(estimator=gnb, param_grid=param_grid)
# grid_search.fit(X, y)

# # Print best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Naive Bayes: Multinomial


# from sklearn.naive_bayes import MultinomialNB
# # Best parameters found:  {'alpha': 0.5}
# # Best cross-validation score: 0.67
# param_grid = {
#     'alpha': [0.1, 0.5, 1.0, 2.0]  # Smoothing parameter
# }

# # Initialize the Gaussian Naive Bayes classifier
# nb = MultinomialNB()

# # Grid search cross-validation
# grid_search = GridSearchCV(estimator=nb, param_grid=param_grid)
# grid_search.fit(X, y)

# # Print best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Stochastic Gradient Descent Classifier

# from sklearn import linear_model
# # param_grid = {
# #     'loss': ['hinge', 'log', 'modified_huber'],
# #     'penalty': ['l2', 'l1'],
# #     'alpha': [0.0001,  0.01],
# #     'max_iter': [1000, 3000],
# #     'learning_rate': ['constant', 'optimal'],
# #     'eta0': [0.01,  0.5]
# # }
# param_grid = {
#     'loss': ['hinge', 'log', 'modified_huber', 'perceptron'],  # Loss function options
#     'alpha': [0.0001, 0.001, 0.01, 0.1]  # Regularization parameter values
# }
# # Best parameters found:  {'alpha': 0.0001, 'loss': 'hinge'}
# # Best cross-validation score: 0.85
# SGDClf = linear_model.SGDClassifier()
# # Grid search cross-validation
# grid_search = GridSearchCV(estimator=SGDClf, param_grid=param_grid)
# grid_search.fit(X, y)

# # Print best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # KNeighbors 


# from sklearn.neighbors import KNeighborsClassifier

# # param_grid = {
# #     'n_neighbors': [3, 5, 9],  # Number of neighbors to use
# #     'weights': ['uniform', 'distance'],  # Weight function used in prediction
# #     'algorithm': ['auto', 'ball_tree', 'kd_tree'],  # Algorithm used to compute the nearest neighbors
# #     'p': [1, 2]  # Power parameter for the Minkowski distance metric
# # }
# # Best parameters found:  {'metric': 'cosine', 'n_neighbors': 11}
# # Best cross-validation score: 0.94
# param_grid = {
#     'n_neighbors': [1,2,3, 5, 7, 9, 11,20,30,40,50,100],  # Number of neighbors options
#     'metric': ['euclidean', 'manhattan', 'cosine']  # Distance metric options
# }
# knn = KNeighborsClassifier()
# # Grid search cross-validation
# grid_search = GridSearchCV(estimator=knn, param_grid=param_grid)
# grid_search.fit(X, y)

# # Print best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Decision Trees 
# from sklearn.tree import DecisionTreeClassifier
# # param_grid = {
# #     'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
# #     'splitter': ['best', 'random'],  # Strategy to choose the split at each node
# #     'max_depth': [None, 5, 15],  # Maximum depth of the tree
# #     'min_samples_split': [2, 10],  # Minimum number of samples required to split a node
# #     'min_samples_leaf': [1, 4],  # Minimum number of samples required at each leaf node
# #     'max_features': ['sqrt', 'log2', None],  # Number of features to consider when looking for the best split
# #     'ccp_alpha': [0.0, 0.1]  # Complexity parameter used for Minimal Cost-Complexity Pruning
# # }
# param_grid = {
#     'max_depth': [None, 5, 10, 15, 20],  # Maximum depth options
#     'criterion': ['gini', 'entropy']  # Criterion options
# }
# # Best parameters found:  {'criterion': 'entropy', 'max_depth': 15}
# # Best cross-validation score: 0.90
# dt = DecisionTreeClassifier()
# # Grid search cross-validation
# grid_search = GridSearchCV(estimator=dt, param_grid=param_grid)
# grid_search.fit(X, y)

# # Print best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Random Forest 

# from sklearn.ensemble import RandomForestClassifier
# # param_grid = {
# #     'n_estimators': [100, 200, 300],  # Number of trees in the forest
# #     'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
# #     'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
# #     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
# #     'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
# #     'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
# #     'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
# # }
# # Best parameters found:  {'max_features': 'log2', 'n_estimators': 300}
# # Best cross-validation score: 0.92
# param_grid = {
#     'n_estimators': [50, 100, 200, 300],  # Number of trees options
#     'max_features': ['sqrt', 'log2', None]  # Maximum features options
# }
# rf = RandomForestClassifier()
# # Grid search cross-validation
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid)
# grid_search.fit(X, y)

# # Print best parameters and score
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


