from train import read_dimacs_directory
import numpy as np


dimacs_directory = './dimacs_files/test_small'

constraints, objective = read_dimacs_directory(dimacs_directory)

c = constraints[0][0]

assignement = [(-1)**(np.random.randint(2))*i for i in range(1,101)]

def get_sat(c, ass):
    sat=0
    for clause in c :
        if len(set(clause) & set(ass)) > 0:
            sat+=1
    return sat

get_sat(c,assignement)
