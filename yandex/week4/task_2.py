import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

close_prices = pd.read_csv("week4-q2\\close_prices.csv")

X = close_prices.iloc[:,1:31]

pca = PCA(n_components=10)

pca.fit(X)

print(pca.explained_variance_ratio_[0:3])

print("3 components:", sum(pca.explained_variance_ratio_[0:3]))
print("4 components:", sum(pca.explained_variance_ratio_[0:4]))

X_pca = pca.transform(X)

djia_index = pd.read_csv("week4-q2\\djia_index.csv")

cc = np.corrcoef(X_pca[:,0],djia_index['^DJI'])

print("corrcoef: ", np.round(cc, 2))

firstComponentMaxCompanyWeight = np.abs(pca.components_[0]).max()
idx = np.abs(pca.components_[0]).argmax()

print("Max weighted company: ",  X.columns[idx])
print("Max weighted company name: ", "V = Visa")