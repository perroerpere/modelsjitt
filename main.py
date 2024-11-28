import pandas as pd
from pygame.transform import threshold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

site1 = pd.read_csv('AA018_Jan_2024.csv', sep=";")
site2 = pd.read_csv('AA018_June_2024.csv', sep=";")
site3 = pd.read_csv('MR102_May_2024.csv', sep=";")
site4 = pd.read_csv('TR041_May_2024.csv', sep=";")


data = pd.concat([site1, site2, site3, site4])

numeric_data = data.select_dtypes(include=['number'])

print(data.head())
print(data.info())

corr = numeric_data.corr()

plt.figure(figsize = (10,8))

sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1,vmax=1)
#plt.show()

data['time'] = pd.to_datetime(data['time'])

print(data['time'])

start_time = data['time'].min()

data['time_seconds'] = (data['time'] - start_time).dt.total_seconds()

print(data['time_seconds'])

threshhold = 5000

data['target'] = (data['battV'] < threshhold).astype(int)

x = data.drop(columns=['battV', 'target', 'loc', 'ip', 'trm', 'time', 'timeUTC'])
y = data['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:" , accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))