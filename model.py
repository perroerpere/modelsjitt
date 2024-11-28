from tokenize import group
import pandas as pd
from numpy.ma.extras import unique
from pandas.core.arrays.timedeltas import sequence_to_td64ns
from pygame.transform import threshold
from scipy.signal.windows import lanczos
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Masking, Input

site1 = pd.read_csv('AA018_Jan_2024.csv', sep=";")
site2 = pd.read_csv('AA018_June_2024.csv', sep=";")
site3 = pd.read_csv('MR102_May_2024.csv', sep=";")
site4 = pd.read_csv('TR041_May_2024.csv', sep=";")
site5 = pd.read_excel('AA018.xlsx')
site6 = pd.read_excel('TR041.xlsx')
site7 = pd.read_excel('MR102.xlsx')


#data = pd.concat([site1, site2, site3, site4])
data = pd.concat([site5, site6, site7])

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



unique_sites = data['loc'].unique()

print(unique_sites)

train_sites, test_sites = train_test_split(unique_sites, test_size=0.2, random_state=42)

train_data = data[data['loc'].isin(train_sites)]
test_data = data[data['loc'].isin(test_sites)]

print(train_data['loc'])
print(test_data['loc'])

print("__________")
print(data.describe())

#data['battV'].hist(bins=50)
#plt.title('distribusjon avv battv')
#plt.xlabel('battv')
#plt.ylabel('frekvens')
#plt.show()

#sns.boxplot(x=data['battV'])
#plt.title('boxplot av battv')
#plt.show()

#data.set_index('time')['battV'].plot(figsize=(10,5))
#plt.title('tidserie av battv')
#plt.show()

data['battV_mean_1y'] = data.groupby('loc')['battV'].transform(
    lambda x:
    x.rolling(window=365*24*12, min_periods=1).mean()
)
print(data['battV_mean_1y'])


'''
data['time_since_last_test'] = data.groupby('loc')['time'].transform(
    lambda x:
    (x-x.shift(1)).dt.total_seconds()
)
print("_________*___")
print(data['time_since_last_test'])
'''

data = data.sort_values(by=['loc', 'time'])

grouped = data.groupby('loc')

sequences = [group[['battV', 'totcur', 'battemp']].values for _,group in grouped]
targets = [group['test'].iloc[-1] for _, group in grouped]

padded_sequences = pad_sequences(sequences, dtype='float32', padding='post', value=0.0)

num_features = padded_sequences.shape[2]

model = Sequential([
    Input(shape=(None, num_features)),
    Masking(mask_value=0.0),
    LSTM(64, return_sequences= False),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

targets = np.array(targets)
model.fit(padded_sequences, targets, epochs=10, batch_size=32)