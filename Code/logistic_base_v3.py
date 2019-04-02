import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

import math

# df = pd.read_pickle('../data/price_level_total_view_2017-01-03_AAPL_grouped.txt') # v1
# df = pd.read_pickle('../data/price_level_total_view_2017-01-03_AAPL_grouped_1') # v2
df = pd.read_pickle('../data/price_level_total_view_2017-01-03_AAPL_grouped_2') # v3
# pd.set_option('display.max_columns', 8)
# print(df)
# list(df)
# len(df) - df.count() # check na's for each feature

def baseline_log(df, train_prop, multi = None):
    # X = df.loc[1:,['mid_price_log', 'trade_volume_differential', 'mid_price_log_differential_0_1', 'mid_price_log_differential_0_2', 'mid_price_log_differential_0_3', 'mid_price_log_differential_1_2', 'mid_price_log_differential_1_3', 'mid_price_log_differential_2_3', 'trade_volume_differential_0_1', 'trade_volume_differential_0_2', 'trade_volume_differential_0_3', 'trade_volume_differential_1_2', 'trade_volume_differential_1_3', 'trade_volume_differential_2_3']]
    # ^ this was for v2 features, below is v3
    X = df.loc[:,['mid_price_log', 'trade_volume_differential', 'mid_price_log_direction_0_1', 'mid_price_log_direction_0_2', 'mid_price_log_direction_0_3', 'mid_price_log_direction_0_4', 'mid_price_log_direction_0_5', 'mid_price_log_direction_0_6', 'trade_volume_differential_direction_0_1', 'trade_volume_differential_direction_0_2', 'trade_volume_differential_direction_0_3', 'trade_volume_differential_direction_0_4', 'trade_volume_differential_direction_0_5', 'trade_volume_differential_direction_0_6']]

    # y = df.loc[1:,'target'] # had to filter out first row in v2, shouldn't in v3
    y = df.loc[:,'target']
    pd.value_counts(y)
    n = X.shape[0]
    cutoff = math.floor(n *train_prop)
    # print('cutoff:',cutoff)

    X_train, X_test = (X.iloc[0:cutoff , :] , X.iloc[cutoff: , :] )
    # print('after x')
    # X_train.tail()
    # X_test.head()

    y_train, y_test = (y[0:cutoff].ravel() , y[cutoff:].ravel() )
    # print('after y')
    # X_train.shape[0] == y_train.shape[0]

    if multi == True:
        logreg = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=500)
        # coeff did not converge for multinomial sag, newton-cg -- lbfgs also gives all one class
    else:
        logreg = LogisticRegression(solver='lbfgs', max_iter=500)

    print('Set-up complete')
    print(np.any(np.isfinite(X_train)), np.any(np.isnan(X_train)))

    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    unique, counts = np.unique(y_pred, return_counts=True)
    print ("\n\n====Value Counts====\n\n", np.asarray((unique, counts)).T,"\n\n==================================")
    print('\n\nAccuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    confuse = confusion_matrix(y_test, y_pred)
    print("\n\n====Confusion Matrix====\n",confuse,"\n\n==================================")
    print(classification_report(y_test, y_pred),"\n\n==================================")

    y_prob = logreg.predict_proba(X_test)
    y_prob = pd.DataFrame(y_prob)
    print(y_prob)
    # y_prob.to_string()




baseline_log(df, train_prop= .8, multi = True)





# get rid of one class (the original caused class imbalance) --> trying to test difference
# df2=df[df['target']!=1]
# df2['target'].value_counts()
# baseline_log(df2, .8, False)

# this gets ~ 89% using the second version of our feature set
# so our model is good at predicting when we are limiting it to either upwards or downwards movement
# --> potentially take an ensemble approach to incoprorate these predictions 