import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier

def import_data(parametr):
    return pandas.read_csv('titanic.csv', index_col = parametr)

def test():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    importances = clf.feature_importances_
    print(importances)




def main():
    data = import_data('PassengerId')
    #taking that rows from dataset
    data_to_let = data.loc[ : ,['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
    #deletind data witn nulls
    data2 = data_to_let.dropna()
    #convert (male,female) -> (0,1)
    SexN = data2.Sex.factorize()
    #adding column
    data2['SexN'] = SexN[0]
    #dataset for tree
    X = data2.loc[ :, ['Pclass', 'Fare' , 'Age', 'SexN']]
    y = data2['Survived']
    #creating clasifier
    clf = DecisionTreeClassifier()
    clf.random_state = 241
    #learning tree
    clf.fit(X,y)
    print(clf)
    print(clf.feature_importances_)
main()






