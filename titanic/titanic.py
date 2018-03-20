#!/usr/bin/python


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import cross_val_score
#import csv as csv

## Now, to get rid of those pesky NaNs...
# ... wait! pandas objects have their own method for interpolating NaNs
# So we don't need an explicit imputer
#from sklearn.preprocessing import Imputer


titanic_dat = pd.read_csv('train.csv' )

## Munge the training data
# Add header names
titanic_filt = titanic_dat.filter([ "PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch" ])
# Encode gender as a Boolean
titanic_filt = titanic_filt.replace("male", 1)
titanic_filt = titanic_filt.replace("female", 0)
# Interpolate missing data (so nice---but have to be careful how this is done)
titanic_filt = titanic_filt.interpolate()

#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

##################################################
## Cross-validation --- always important
##################################################
#
#cv_scores_100 = cross_val_score(RandomForestClassifier(n_estimators=100, n_jobs=1), X=titanic_filt.drop(["PassengerId", "Survived"], axis=1).values, y=titanic_filt.loc[:, 'Survived'].values, cv=10)
#print("Cross val scores for n=100: " + str(cv_scores_100) )
#print("Mean of cross-val scores (n=100): " + str( np.mean(cv_scores_100)) )
#
# Well, not *too* bad at about 80%:
#Mean of cross-val scores (n=100): 0.795887810691
#
##################################################


# Create the model
rf = RandomForestClassifier(n_estimators=100, n_jobs=1).fit(X=titanic_filt.drop(labels='Survived', axis=1).values, y=titanic_filt.loc[:, 'Survived'].values)

# Test the model
test_dat = pd.read_csv("test.csv")

# Munge the test data. Note that this should really be a function to reduce
# errors replicating what happens with the training set.
test_filt = test_dat.filter([ "PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch" ])
test_filt = test_filt.replace("male", 1)
test_filt = test_filt.replace("female", 0)
test_filt = test_filt.interpolate()

output = rf.predict(test_filt)

# Put the output in a format the Kaggle competition wants
output_str=zip(test_filt.filter("Passenger"),",",output)

np.savetxt(fname="output_test.csv",X=zip(test_filt["PassengerId"].values,output), delimiter=",")
