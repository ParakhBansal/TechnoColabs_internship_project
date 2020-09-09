#let's import basic modules here
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

#now let's read the training data
df = pd.read_csv('parkinsons.data')
df.head()

#this is a plot of target column
explode = (0,0.3)
plt.pie(df.status.value_counts(),labels=['1','0'],autopct='%.2f',explode=explode)

#preparing of data
y = df['status']
X = df.drop(['status','name'], axis = 1)

#model instance
model = XGBClassifier(random_state=7)

#fitting model
model.fit(X.values,y.values)

pickle.dump(model, open('xgbmodel.pkl','wb'))
