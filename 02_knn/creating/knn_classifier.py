from pandas import DataFrame
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# Creating Train-test dataset
X_train, Y_train = make_blobs(
    n_samples=300, centers=2, n_features=2, cluster_std=6, random_state=11)

# Plotting the data
df = DataFrame(dict(x=X_train[:, 0], y=X_train[:, 1], label=Y_train))
colors = {0: 'blue', 1: 'orange'}
fig, ax = plt.subplots(figsize=(8, 8))
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x',
               y='y', label=key, color=colors[key])
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.show()

# Feature Scaling
def normalize(X):
    x1_min = min(X_train[:, 0])
    x1_max = max(X_train[:, 0])

    def f(x): return (x - x1_min)/(x1_max - x1_min)
    X[:, 0] = f(X[:, 0])

    x2_min = min(X_train[:, 1])
    x2_max = max(X_train[:, 1])

    def f(x): return (x - x2_min)/(x2_max - x2_min)
    X[:, 1] = f(X[:, 1])

    return X

X = normalize(X_train)
print(X[0:5])

# Heart of KNN
def find_neighbors(k, X_tr, new_point):
    neighbor_arr = []
    for i in range(len(X_tr)):
        dist = np.sqrt(sum(np.square(X_tr[i]-new_point)))
        neighbor_arr.append([i, dist])
    neighbor_arr = sorted(neighbor_arr, key = lambda x : x[1])
    
    return neighbor_arr[0:k]

# Classification with KNN
from collections import Counter
def classifier(neighbor_arr):
    class_arr = [Y_train[i[0]] for i in neighbor_arr]
    return Counter(class_arr).most_common(1)[0][0]

new_points = np.array([[-10, -10],
                      [0, 10],
                      [-15, 10],
                      [5, -2]])

new_points = normalize(new_points)

knn = find_neighbors(4, X, new_points[0])
print(classifier(knn))

knn = find_neighbors(4, X, new_points[1])
print(classifier(knn))

knn = find_neighbors(4, X, new_points[2])
print(classifier(knn))

knn = find_neighbors(4, X, new_points[3])
print(classifier(knn))