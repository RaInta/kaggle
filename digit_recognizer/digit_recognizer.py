#!/usr/bin/python


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import cross_val_score

# Pandas saves so much time cleaning up imported CSV files....
mnist_data = pd.read_csv('train.csv' )

##################################################
# This next section is for cross-validation only
##################################################
#
#cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=-1),
#                            X=mnist_data.drop('label', axis=1).values,
#                            y=mnist_data.loc[:, 'label'].values,
#                            cv=10)
#
## Cross-validation stage
#cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=1), X=dat.drop('label', axis=1).values, y=dat.loc[:, 'label'].values, cv=10)
#print("Cross val scores: " + str(cv_scores) )
##################################################


# This is the model. Isn't it cute how it's a single line?
rf = RandomForestClassifier(n_estimators=100, n_jobs=1).fit(X=mnist_data.drop(labels='label', axis=1).values, y=mnist_data.loc[:, 'label'].values)

# Test data
test = pd.read_csv("test.csv")

# Apply the model
output = rf.predict(test)

# Create the output
np.savetxt(fname="output_test.csv",X=output, delimiter="\n")
