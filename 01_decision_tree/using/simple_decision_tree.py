from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sep-alt', 'sep-comp', 'pet-alt', 'pet-comp', 'class']

dataset = pd.read_csv(url, names=names)

#print(dataset)

#train = dataset[:120]
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

print(tree.plot_tree(clf))
#y = Y_test[0]
#y = y.reshape(1, -1)

#print(y)
#print(X_test[0])

#print(clf.predict(X_test[0], y))

# print(result)

#print(train)

#X = [[0, 0], [1, 1]]
#Y = [0, 1]
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X, Y)

#print(clf)

#print(clf.predict([[2., 2.]]))