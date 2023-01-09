# import libraries
import os, pickle,datetime
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, Input
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import LSTM, Dropout, Dense

# define file paths
train_csv_path = os.path.join(os.getcwd(),'Data','cases_malaysia_train.csv')
test_csv_path = os.path.join(os.getcwd(),'Data','cases_malaysia_test.csv')
save_path = os.path.join(os.getcwd(),'Models', 'Model.h5')

# load data
df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)

# inspect train data
df_train.info() # cases_new type object -> need to convert to float64
df_train.describe().T
df_train.isna().sum() # NaNs in 2020 on cluster_* columns
msno.bar(df_train)

# inspect test data
df_test.info() # cases_new missing 1 value
df_test.describe().T
df_test.isna().sum()
msno.bar(df_test)

# data cleaning/filling
# convert cases_new to float64
df_train['cases_new'] = pd.to_numeric(df_train['cases_new'], errors='coerce')
df_train.isna().sum()

# new missing values in cases_new
# impute missing data in both datasets using interpolation
df_train['cases_new'].interpolate(method='linear',inplace=True)
df_test['cases_new'].interpolate(method='linear',inplace=True)

# data visualization
fig = plt.figure(figsize=[15,8])
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)
df_train['cases_new'].plot.line(ax=ax1)
df_test['cases_new'].plot.line(ax=ax2)
ax1.set_title('Covid-19 Daily New Cases (Dec 2021 - Mar 2022)')
ax1.set_ylabel('New cases')
ax1.set_xlabel('Days since 25 Jan 2020')
ax2.set_title('Covid-19 Daily New Cases (Jan 2020 - Dec 2021)')
ax2.set_ylabel('New cases')
ax2.set_xlabel('Days since 5 Dec 2021')
plt.tight_layout()
plt.show()

# feature selection
X = df_train['cases_new']
# normalize
mms = MinMaxScaler()
X = mms.fit_transform(np.expand_dims(X,axis=-1))

# save after fit_transform
MMS_PATH = os.path.join(os.getcwd(), 'Models', 'mms.pkl')
with open(MMS_PATH, 'wb') as file:
    pickle.dump(mms,file)

# segregate data
win_size = 30
X_train = []
y_train = []

for i in range(win_size,len(X)):
    X_train.append(X[i-win_size:i])
    y_train.append(X[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# prepare test data in the same way but after combine
combined = pd.concat((df_train['cases_new'],df_test['cases_new']))
num_days = win_size +len(df_test)
total = combined[-num_days:]
test_temp = mms.transform(np.expand_dims(total,axis=-1))

X_test = []
y_test = []

for i in range(win_size,len(test_temp)):
    X_test.append(test_temp[i-win_size:i])
    y_test.append(test_temp[i])

X_test = np.array(X_test)
y_test = np.array(y_test)

# create model layers
model = Sequential()
model.add(Input(shape=(np.shape(X_train)[1:])))
model.add(LSTM(64,return_sequences=(True),activation='relu'))
model.add(Dropout(0.3))
model.add(LSTM(64,return_sequences=(True),activation='relu'))
model.add(Dropout(0.3))
model.add(LSTM(64,activation='relu'))
model.add(Dense(1))
model.summary()
# plot model for reference
plot_model(model,show_shapes=True,show_layer_names=True)

# train model
model.compile(optimizer='adam',loss='mse',
              metrics=['mean_absolute_percentage_error','mse'])
log_path = os.path.join(os.getcwd(),'log_dir',
            datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tb = TensorBoard(log_dir=log_path)
es = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)
hist = model.fit(X_train,y_train,epochs=100,
                    callbacks=[tb,es],validation_data=(X_test,y_test))

# plot error graphs
plt.figure(figsize=(15,15))
for hist_key,subplot_no in zip(hist.history.keys(),[1,2,3,4,5,6]):
    plt.subplot(2,3,subplot_no)
    plt.plot(hist.history[hist_key])
    plt.xlabel(hist_key)
plt.show()

# predict
y_pred = model.predict(X_test)

# plot predictions
plt.figure()
plt.plot(y_test,color='red')
plt.plot(y_pred,color='blue')
plt.xlabel('Time')
plt.ylabel('Cases new')
plt.legend(['Actual', 'Predicted'])
plt.show()

# calc mape
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE (Mean Absolute Percentage Error) is {np.round(mape*100,2)}%')
model.save(save_path)