# Cody Herberholz
# CS445 HW3 Support Vector Machines

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

data = pd.read_csv("spambase.data", header=None, index_col=57)
print(data)
data = utils.shuffle(data)

x_train, x_test, y_train, y_test = train_test_split(data, data.index.values, test_size=0.5)

sc = StandardScaler().fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

# Change 0s to -1s
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

# Train SVM
svm_new = svm.SVC(kernel='linear', C=1.0, random_state=0)
svm_new.fit(x_train, y_train)

y_score = svm_new.decision_function(x_test)
y_predict = svm_new.predict(x_test)

print(metrics.accuracy_score(y_test, y_predict))
print(metrics.precision_score(y_test, y_predict))
print(metrics.recall_score(y_test, y_predict))
# print(metrics.classification_report(y_test, y_predict))

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr, tpr)

# print("FPR: " , fpr)
# print("TPR: " , tpr)
# print("Thresholds: " ,thresholds)

# Plot ROC Curve
plt.title('ROC')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-.1, 1.1])
plt.ylim([-.1, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()