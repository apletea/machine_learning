import numpy as np
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import  DictVectorizer

def one_hot_ex(data_train, data_test):
    enc = DictVectorizer()
    data_train['LocationNormalized'].fillna('nan', inplace=True)
    data_train['ContractTime'].fillna('nan', inplace=True)
    X_train_cated = enc.fit_transform(data_train[['LocationNormalized','ContractTime']].to_dict('records'))
    X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

def main():
    data_train = pandas.read_csv('salary-train.csv')
    print(data_train)
main()