import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
train_data = np.load('extra_feature_matrices/train_examples_5000sp.npz')
X = train_data['inputs']
y = train_data['targets']

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=1000,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
LABELS = np.array(["x pos", "y pos", "size-x", "size-y", "avg R", "avg G", "avg B", "var R", "var G", "var B", "avg H", "avg S", "avg V", "avg entropy", "edge freq"])
LABELS_sorted = LABELS[indices]


plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), LABELS_sorted)
plt.xlim([-1, X.shape[1]])
plt.show()