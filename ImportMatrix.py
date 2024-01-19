# Disable numpy threading to get useful timings
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Import Libraries
import numpy as np
from numpy.linalg import norm
import pandas as pd
import math as mt
import csv

# Define the function to load the matrix of YearPredictionMSD data set
def read_matrixY(n,c):
    matrix=[]

    with open('YearPredictionMSD.txt', 'r') as file:
        lines = [next(file) for _ in range(n)]

    for line in lines:
        valuess=line.split('  ')[1]
        elements=valuess.split()
        values=[elem.split(':')[1] for elem in elements]
        matrix.append(values)

    A = np.zeros((n, n), dtype='d')

    for i in range(len(matrix)):
        x_i = np.array(matrix[i], dtype='d')
        for j in range(i, len(matrix)):
            x_j = np.array(matrix[j], dtype='d')
            v = mt.exp(-norm(x_i - x_j)**2 / c**2)
            A[i, j] = v
            A[j, i] = v

    np.fill_diagonal(A, 1.0)
    return A

# Define the function to load the matrix of mnist data set
def read_matrixM(n, c):
    with open('mnist.scale.txt', 'r') as file:
        lines = [next(file) for _ in range(n)]

    data = []
    max_index = 0

    for line in lines:
        elements = line.split()
        values = {int(elem.split(':')[0]): float(elem.split(':')[1]) for elem in elements[1:]}
        max_index = max(max_index, max(values.keys()))
        for idx in range(780):
            if idx not in values:
                values[idx] = 0.0
        data.append(values)

    df = pd.DataFrame(data)
    df = df.reindex(columns=range(max_index), fill_value=0.0)

    A = np.zeros((n, n), dtype='d')

    for i in range(n):
        row_i = df.iloc[i, :]
        x_i = row_i.to_numpy(dtype='d')
        for j in range(i, n):
            row_j = df.iloc[j, :]
            x_j = row_j.to_numpy(dtype='d')
            v = mt.exp(-(norm(x_i - x_j) / c)**2)
            A[i, j] = v
            A[j, i] = v

    np.fill_diagonal(A, 1.0)
    return A

# Read the matrix with dimension m
matrix = read_matrixY(4096,100000)
# matrix = read_matrixY(4096,10000)
# matrix = read_matrixM(4096, 100)

# Wite the matrix in the .csv file
with open('matrix4096Y105.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(matrix)