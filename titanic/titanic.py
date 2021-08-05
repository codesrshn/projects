import re
import pandas as pd
import csv
from datetime import date
import sys
import pprint
from sklearn.linear_model import LogisticRegression

passengers = pd.read_csv('train.csv',sep=',')
impt = ['PassengerId','Pclass','Parch']
passnger = passengers.loc[:,impt]

saved = passengers.Survived 
trainsurvived = LogisticRegression()
trainsurvived.fit(passnger,saved)

testdata = pd.read_csv('test.csv',sep=',')
testpass = testdata.loc[:,impt]
survivePrediction = trainsurvived.predict(testpass)

#pprint.pprint(passnger)

traineddata = pd.DataFrame({'PassengerId':testpass.PassengerId,'Survived':survivePrediction}).set_index('PassengerId')
traineddata.to_csv('Passenger_Survival_Report.csv')
