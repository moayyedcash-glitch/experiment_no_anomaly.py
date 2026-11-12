import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

np.random.seed(48)
num_processes = 100

arrival_time = np.random.randint(0, 100, num_processes)
burst_time = np.random.randint(1, 30, num_processes)

labels = np.array([1 if bt > 15 else 0 for bt in burst_time])

data = pd.DataFrame({
    'arrival': arrival_time,
    'burst': burst_time,
    'label': labels
})

X = data[['arrival','burst']].values
y = labels

X = X.reshape((X.shape[0], 1, X.shape[1]))

model = Sequential()
model.add(LSTM(32, input_shape=(1,2)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, verbose=1)

pred = model.predict(X)
predicted_labels = (pred > 0.5).astype(int).flatten()

data['predicted'] = predicted_labels

data_fcfs = data.sort_values(by='arrival')
waiting_time_fcfs = []
current_time = 0

for _, row in data_fcfs.iterrows():
    if current_time < row['arrival']:
        current_time = row['arrival']
    waiting_time_fcfs.append(current_time - row['arrival'])
    current_time += row['burst']

data_fcfs['waiting_time'] = waiting_time_fcfs

data_smart = data.sort_values(by=['predicted','arrival'])
waiting_time_smart = []
current_time = 0

for _, row in data_smart.iterrows():
    if current_time < row['arrival']:
        current_time = row['arrival']
    waiting_time_smart.append(current_time - row['arrival'])
    current_time += row['burst']

data_smart['waiting_time'] = waiting_time_smart

fcfs_mean = data_fcfs['waiting_time'].mean()
smart_mean = data_smart['waiting_time'].mean()

print("FCFS:", fcfs_mean)
print("Smart:", smart_mean)

plt.bar(['FCFS','Smart'], [fcfs_mean, smart_mean])
plt.show()

with open('results.txt', 'w') as f:
    f.write(f"FCFS: {fcfs_mean}\n")
    f.write(f"Smart: {smart_mean}\n")

improvement = ((fcfs_mean - smart_mean) / fcfs_mean) * 100
print(f"Percentage Improvement: {improvement:.2f}%")
#plt.show()
