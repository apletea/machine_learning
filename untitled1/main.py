import pandas
import os
from sets import  Set

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import roc_curve,roc_auc_score

labeled_data = pandas.read_csv('train.csv', index_col='Id')
train_img_data = pandas.read_csv('img_train.csv', index_col='Id')
test_data = pandas.read_csv('test.csv',index_col='Id')
test_img_data = pandas.read_csv('img_test.csv',index_col='Id')

def abra_cadabra(parameter, data):
    something = Set()
    something_a= []
    for val in data[parameter]:
        something.add(val)
        something_a.append(val)
    something_le = preprocessing.LabelEncoder()
    something_le.fit(list(something))
    something_a = something_le.transform(something_a)
    return something_a



languages_a = abra_cadabra('Language',labeled_data)
countries_a = abra_cadabra('Country',labeled_data)
ratings_a = abra_cadabra('Rating',labeled_data)



labeled_data['Language'],labeled_data['Country'],labeled_data['Rating'] = languages_a,countries_a,ratings_a

data = labeled_data.iloc[0:3635, 1:27]
data['Poster'] = train_img_data['Prob']

#data = preprocessing.scale(data)
labels = labeled_data.iloc[0:3635, :1]

clf = MLPClassifier(activation='relu',batch_size=255,random_state=247)
#clf = DecisionTreeClassifier(random_state=247)
#clf = svm.SVC(random_state=247)
#clf = QuadraticDiscriminantAnalysis()
#clf = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
clf.fit(data,labels.values.ravel())
print clf.predict_proba(test_data)[:,:1]
print clf.predict(test_data)
#print roc_auc_score(test_labeles,clf.predict(test_data))