import numpy as np
import pandas
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn import cross_validation

def main():
    #load data set
    set = load_boston()
    #define X and y
    X = set['data']
    y = set['target']
    #seting to equals
    attributes =scale(X)
    #cross validator
    kfolgen = KFold(n = y.shape[0],n_folds=5,shuffle=True,random_state=42)

    for p in np.linspace(1, 10, num=200):
        #clasifier
        clf = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
        #cross validation score
        sc = cross_validation.cross_val_score(clf, X, y, cv=kfolgen, scoring='mean_squared_error')
        #average
        msc = np.mean(sc)
        #print average
        print(msc)

main()