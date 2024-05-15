# Importing the required libraries
import numpy as np
import os


def dropLast():
    directory = os.getcwd()+"\data"
    files = os.listdir(directory)

    for file in files:
        path = directory + f'\{file}'
        data = np.loadtxt(path)
        # edited_data = data[:, :-15]
        for i in range(25):
            data = np.delete(data,9,1)
        print(np.shape(data))
        np.savetxt('temp.txt', data)  # Save edited data to a temporary file called 'temp_data.txt'
        os.replace('temp.txt', path)  # Replace original file with temporary file
