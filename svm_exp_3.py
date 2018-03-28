# Cody Herberholz
# CS445 HW3 Support Vector Machines
# Experiment 3: random features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn import svm
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read-in/process data
data = pd.read_csv("spambase.data", header=None, index_col=57)
data = utils.shuffle(data)

x_train, x_test, y_train, y_test = train_test_split(data, data.index.values, test_size=0.5)

sc = StandardScaler().fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

# Change 0s to -1s
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

# Train SVM
svm_old = svm.SVC(kernel='linear', C=1.0, random_state=0)

n_features = 57
y_scores = []

# initalizes y_scores list
for i in range(56):
    y_scores.append(0)

# starts at 57 features and then randomly deletes one feature at a time
# until features reach 2
while n_features >= 2:
    num = randint(0, n_features-1)
    svm_old.fit(x_train, y_train)
    y_scores[n_features - 2] = svm_old.score(x_test, y_test)
    x_train = np.delete(x_train, num, 1)
    x_test = np.delete(x_test, num, 1)
    n_features -= 1

# Plot ROC Curve
plt.title('Random Features vs. Accuracy')
plt.plot(y_scores)
plt.ylabel('Accuracy')
plt.xlabel('Features')
plt.show()