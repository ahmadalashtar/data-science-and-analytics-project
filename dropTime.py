# Importing the required libraries
import numpy as np
import os


def dropTime():
    directory = os.getcwd()+"\data"
    files = os.listdir(directory)

    for file in files:
        path = directory + f'\{file}'
        data = np.loadtxt(path)
        # edited_data = data[:, :-15]
        data = np.delete(data,0,1)
        data = np.delete(data,0,1)
        data = np.delete(data,0,1)
        data = np.delete(data,0,1)
        print(np.shape(data))
        np.savetxt('temp.txt', data)  # Save edited data to a temporary file called 'temp_data.txt'
        os.replace('temp.txt', path)  # Replace original file with temporary file
