# Cody Herberholz
# CS445 HW3 Support Vector Machines
# Experiment 2: Chosen Features

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

# read-in/preprocessing of data
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

n_features = 2
y_scores = []

# initializes y_scores list
for i in range(56):
    y_scores.append(0)

# uses RFE to choose two best features and using a loop increase number
# of features by one until 57 features have been reached.
while n_features <= 57:
    svm_old.fit(x_train, y_train)
    selector = RFE(svm_old, n_features, step=1)
    new_x_train = selector.fit_transform(x_train, y_train)
    new_x_test = selector.fit_transform(x_test, y_test)
    svm_old.fit(new_x_train, y_train)
    y_scores[n_features - 2] = svm_old.score(new_x_test, y_test)
    n_features += 1

# Plot ROC Curve
plt.title('Features vs. Accuracy')
plt.plot(y_scores)
plt.ylabel('Accuracy')
plt.xlabel('Features')
plt.show()