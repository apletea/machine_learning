import pandas
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold
from sklearn import cross_validation

def ex():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([-3, 1, 10])
    clf = RandomForestRegressor(n_estimators=100)
    clf.fit(X, y)
    predictions = clf.predict(X)
    print(r2_score([10, 11, 12], [9, 11, 12.1]))


def main():
    data = pandas.read_csv('abalone.csv')
    data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x=='F' else 0))

    X = data.iloc[:,0:8]
    y = data.iloc[:,8]

    kfoldGen = KFold(n=y.shape[0],n_folds=5,shuffle=True,random_state=1)
    res = pandas.DataFrame(columns=['k','score'])

    for k in range(0,51):
        clf = RandomForestRegressor(n_estimators=k,random_state=1)
        sc = cross_validation.cross_val_score(clf,X,y,cv=kfoldGen,scoring='r2')
        msc = np.mean(sc)
        res.loc[k] = [k,msc]

    with open('ans.txt', 'w') as f1:
        f1.write(str(int(res.query('score > 0.52')['k'].min())))
    print(data)
main()