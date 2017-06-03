import numpy as np
import pandas
from sklearn.svm import SVC




def main():
    data = pandas.read_csv('svm-data.csv', header=None)

    y=data[0]
    X = data.loc[:,1:2]
    #SVC clasiffier C
    clf = SVC(C=100000, random_state=241)
    clf.fit(X,y)

    print(clf.support_vectors_)
    print(clf.support_)
main()