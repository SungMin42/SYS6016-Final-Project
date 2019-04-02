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

# get rid of one class (the unchanged caused class imbalance) --> trying to test difference
df2=df[df['target']!=1]
df2['target'].value_counts()

def baseline_log(df, multi = None):
    X = df.iloc[:,0:2]
    y = df.iloc[:,2:3]
    pd.value_counts(y['target'])
    n = X.shape[0]
    cutoff = n-(n//8) # total - the number you want to test, which here i'm flooring 
    #                   (amount you want in training should be 1/10th value the denominator)

    X_train, X_test = (X.iloc[0:cutoff , :] , X.iloc[cutoff: , :] )
    # X_train.tail()
    # X_test.head()

    y_train, y_test = (y.iloc[0:cutoff , :].values.ravel() , y.iloc[cutoff: , :].values.ravel() )

    # X_train.shape[0] == y_train.shape[0]

    if multi == True:
        logreg = LogisticRegression(multi_class='ovr', solver='lbfgs')
        # coeff did not converge for multinomial sag, newton-cg -- lbfgs also gives all one class
    else:
        logreg = LogisticRegression(solver='lbfgs')

    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    unique, counts = np.unique(y_pred, return_counts=True)
    print (np.asarray((unique, counts)).T)
    print('\n\nAccuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    confuse = confusion_matrix(y_test, y_pred)
    print("\n\n\n===Confusion Matrix===\n",confuse,"\n\n")
    print(classification_report(y_test, y_pred))

    y_prob = logreg.predict_proba(X_test)
    y_prob = pd.DataFrame(y_prob)
    print(y_prob)
    # y_prob.to_string()




baseline_log(df, True)
baseline_log(df2, False)