from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
iris = datasets.load_iris()
list(iris.keys())
X = iris['data'][:, 3:]
y = (iris['target'] == 2).astype(np.int)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print(X)
print(y)
print(X_new)