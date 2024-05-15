# Importing the required libraries
import numpy as np
import os
import random


def missingData():
    directory = os.getcwd()+"\data"
    files = os.listdir(directory)

    for file in files:
        rowCount = 0
        path = directory + f'\{file}'
        data = np.loadtxt(path)
        numRows = np.shape(data)[0]
        while(numRows<426):
           row = random.randrange(numRows)
           data = np.insert(data, row, data[row,:], axis=0)
           numRows += 1
        print(np.shape(data))
        np.savetxt('temp.txt', data)  # Save edited data to a temporary file called 'temp_data.txt'
        os.replace('temp.txt', path)  # Replace original file with temporary file
