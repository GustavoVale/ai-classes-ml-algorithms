from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
from decision_tree_classifier import DecisionTreeClassifier


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sep-alt', 'sep-comp', 'pet-alt', 'pet-comp', 'class']
dataset = pd.read_csv(url, names=names)

# print(dataset.head(10))

# Train-test split
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.2, random_state=41)

# Fit the model
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train,Y_train)
classifier.print_tree()

# Test the model
Y_pred = classifier.predict(X_test) 
print(accuracy_score(Y_test, Y_pred))
