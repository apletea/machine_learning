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
    data_test = pandas.read_csv('salary-test-mini.csv')

    data_train['FullDescription'] = data_train['FullDescription'].str.lower()
    data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
    data_train['LocationNormalized'].fillna('nan', inplace=True)
    data_train['ContractTime'].fillna('nan', inplace=True)

    data_test['FullDescription'] = data_test['FullDescription'].str.lower()
    data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    data_test['LocationNormalized'].fillna('nan', inplace=True)
    data_test['ContractTime'].fillna('nan', inplace=True)

    vectorizer = TfidfVectorizer(min_df=5)
    X_train_fd_vect = vectorizer.transform(data_train['FullDescription'])
    X_test_fd_vect = vectorizer.transform(data_test['FullDescription'])

    enc = DictVectorizer()
    X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
    X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

    X_train = np.hstack([X_train_fd_vect, X_train_categ])
    X_test = np.hstack([X_test_fd_vect, X_test_categ])

    r = Ridge(alpha=1)
    r.fit(X_train, data_train['SalaryNormalized'])

    p = r.predict(X_test)

    print("Predicted salary: ", " ".join(map(str, np.round(p, 2))))
main()