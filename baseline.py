# Importing the required libraries
import numpy as np
import os


def removeBaseline():
    directory = os.getcwd()+"\data"
    files = os.listdir(directory)

    for file in files:
        rowCount = 0
        path = directory + f'\{file}'
        data = np.loadtxt(path)
        for row in data:
            if row[0]< 20 or row[0] > 80:
                data = np.delete(data,rowCount,0)
                rowCount -= 1
            rowCount += 1
        print(np.shape(data))
        np.savetxt('temp.txt', data)  # Save edited data to a temporary file called 'temp_data.txt'
        os.replace('temp.txt', path)  # Replace original file with temporary file
