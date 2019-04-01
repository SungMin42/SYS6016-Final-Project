# baseline model: chose logistic because of required train time for our dataset if we were to stick with SVM

# see below link to get started using sklearn
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

df = pd.read_pickle('../data/price_level_total_view_2017-01-03_AAPL_grouped.txt')
df.head()
df.tail()

X = df.iloc[:,0:2]
y = df.iloc[:,2:3]
pd.value_counts(y['target'])

n = X.shape[0]
cutoff = n-(n//8) # total - the number you want to test, which here i'm flooring 
#                   (amount you want in training should be 1/10th value the denominator)
# cutoff

X_train, X_test = (X.iloc[0:cutoff , :] , X.iloc[cutoff: , :] )
X_train.tail()
X_test.head()

y_train, y_test = (y.iloc[0:cutoff , :].values.ravel() , y.iloc[cutoff: , :].values.ravel() )

# X_train.shape[0] == y_train.shape[0]



logreg = LogisticRegression(multi_class='ovr')
# logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs') # coeff did not converge for sag, newton-cg
                                                                    # lbfgs also gives all one class
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
unique, counts = np.unique(y_pred, return_counts=True)
print('\n\n\n\n\n')
print (np.asarray((unique, counts)).T)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))




confusion_matrix = confusion_matrix(y_test, y_pred)
print("===Confusion Matrix===\n",confusion_matrix,"\n\n")

# throws an error because our multi-class log doesn't predict for each label --> basically just predicts majority class
# print(classification_report(y_test, y_pred))

# ================================================
# cannot multi-class

# logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
# plt.figure()
# plt.plot(false_positive_rate, true_positive_rate, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.savefig('Log_ROC')
# plt.show()