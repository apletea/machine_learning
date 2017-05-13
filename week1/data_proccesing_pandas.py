import pandas
import numpy as np


#generaation of random matrix 1000 rows and 50 columns numbers are from 1 to 100
def random_matrix(_loc, _scale, _columns, _rows):
    X = np.random.normal(loc=_loc,scale=_scale,size=( _columns, _rows))
    return X

#substraction from each column its average, and divider it on standart norm
def normalize_matrix(X):
    m = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = ((X - m) / std)
    return X_norm

#returns the numbers of rows where summ of elements more than target
def rows_with_mostly(M ,target):
    r = np.sum(M, axis=1)
    return np.nonzero( r > target)

#return merge of two matrisec
def merger_matrix(A,B):
    AB = np.vstack((A, B))
    return AB

def import_data(keyWord):
    data = pandas.read_csv('titanic.csv', index_col=keyWord)
    return data

#counts male in ship
def count_male(data):
    ans = 0
    for str in data['Sex']:
        if (str == 'male'):
            ans+=1
    return ans

def main():
    data = import_data('PassengerId')
    print(count_male(data))
    print(len(data['Sex'])-count_male(data))
main()
