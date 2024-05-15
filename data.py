# Importing the required libraries
import numpy as np
import os

def getData():
    labels = ['nothin', 'Acetone', 'Ethanol','Gin','Methane']
    X = []
    y = []
    labelIndex = 0
    counter = 0


    directory = os.getcwd()+"\data"
    files = os.listdir(directory)

    for file in files:
        path = directory + f'\{file}'
        counter += 1
        data = np.loadtxt(path)
        
        index = 0
        if file[0] == 'A':
            index = 1
        elif file[0] == 'E':
            index = 2
        elif file[0] == 'G':
            index = 3
        elif file[0] == 'M':
            index = 4
        for row in data:
            X.append(row)
            y.append(index)

    X = np.array(X)
    y = np.array(y)
    return X, y