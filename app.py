import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Price Prediction')
link = '[Get Stock Ticker](https://finance.yahoo.com/)'
st.markdown(link, unsafe_allow_html=True)


user_input = st.text_input('Enter Stock Ticker','AAPL')
df = data.DataReader(user_input,'yahoo',start,end)

#describing data
st.subheader('DAta from 2010-2019')
st.write(df.describe())

#visualisation
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
#Splitting data into x train and y train


#Loading the model
model = load_model('keras_model.h5')
#testing
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final graph
st.subheader('Prediction vs original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = "Original Price")
plt.plot(y_predicted, 'r', label = "predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


test_df = pd.DataFrame(y_test)
predicted_df = pd.DataFrame(y_predicted)
df1 = data.DataReader('AAPL','yahoo',start,end)
df2 = df1.rename(columns={'Date': 'DATE'})
df2 = df2.drop(['Volume'], axis = 1)
df2 = df2.filter(['Date'],axis = 1)
df2.to_csv('date.csv')
df3 = pd.read_csv('date.csv')
df3 = df3.filter(['Date'])

date_testing = pd.DataFrame(df3['Date'][int(len(df3)*0.70): int(len(df3))])
test_df.to_csv('testdat.csv')
pd4 = pd.read_csv('testdat.csv')
pd4.filter(['0'])
pd4 = pd4.rename(columns={'0':'Actual Price'})
pd4 = pd4.filter(['Actual Price'])
pd4.to_csv('Actual.csv',index=False)

predicted_df.to_csv('predicteddata.csv')
df5 = pd.read_csv('predicteddata.csv',index_col = False)
df5.filter(['0'])
df5 = df5.rename(columns={'0':'Predicted Price'})
df5 = df5.filter(['Predicted Price'])
df5.to_csv('Predicted.csv',index=False)



date_testing = date_testing.reset_index()
result1 = pd.concat([date_testing,pd4],axis = 1)
result2 = pd.concat([result1,df5],axis =1)
result2 = result2.filter(['Date','Actual Price','Predicted Price'])
result2.to_csv('final.csv',index=False)
st.subheader('Actual Price vs Predicted Price')
df1 = pd.read_csv('final.csv')
st.write(df1)
