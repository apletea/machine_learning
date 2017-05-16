import numpy as np
import pandas
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import preprocessing






def main():
    #dowload data with out headers
    data = pandas.read_csv('wine.data', header = None)
    #data
    class_wine = data[0]
    winde_data = data.loc[:, 1:13]
    #crossvalidation k - neighbors
    kfolden = KFold(n=class_wine.shape[0], n_folds = 5, shuffle=True,random_state=42)

    print ('k    score')
    #scale data to equal diaposons(for the same weight for each attribute
    winde_data = preprocessing.scale(winde_data)
    for k in range(1,51):
        # define classifier
        clf = KNeighborsClassifier(n_neighbors=k)
        #cross validation score
        sc = cross_validation.cross_val_score(clf,winde_data,class_wine, cv=kfolden)
        #finding average
        msc = np.mean(sc)
        print(k,   msc)
main()