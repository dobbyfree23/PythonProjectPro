#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pandas와 numpy 라이브러리를 가져옵니다.
import pandas as pd
import numpy as np

# matplotlib과 seaborn 라이브러리를 가져옵니다.
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn의 스타일을 지정합니다.
sns.set_style('whitegrid')

# matplotlib의 스타일을 지정합니다.
plt.style.use("fivethirtyeight")

# 주피터 노트북에서 그래프를 볼 수 있도록 합니다.
# get_ipython().run_line_magic('matplotlib', 'inline')

# yahoo에서 주식 데이터를 읽어오기 위해 pandas_datareader의 DataReader와 yfinance를 import합니다.
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

# yfinance를 설정합니다.
yf.pdr_override()

# 시간 데이터를 다루기 위해 datetime 모듈을 import합니다.
from datetime import datetime

# 분석할 기업 리스트를 정의합니다.
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

# 데이터를 가져올 기간을 설정합니다.
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# 각 기업의 주식 데이터를 다운로드하고 변수에 저장합니다.
for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)

# 각 기업에 대한 정보를 담고 있는 리스트를 정의합니다.
company_list = [AAPL, GOOG, MSFT, AMZN]
# 각 기업의 이름을 담고 있는 리스트를 정의합니다.
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

# 각 기업의 주식 데이터에 company_name 컬럼을 추가합니다.
for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name

# 모든 기업의 주식 데이터를 합쳐서 하나의 데이터프레임으로 만듭니다.
df = pd.concat(company_list, axis=0)

# 데이터프레임의 마지막 10개 행을 출력합니다.
df.tail(10)


# In[2]:


# AAPL 변수에 저장된 애플 주식 데이터의 요약 통계 정보를 출력합니다.
AAPL.describe()


# In[3]:


# AAPL 변수에 저장된 애플 주식 데이터의 일반 정보를 출력합니다.
AAPL.info()


# In[4]:


# 그래프의 크기를 지정합니다.
plt.figure(figsize=(15, 10))

# 그래프 간의 간격을 조절합니다.
plt.subplots_adjust(top=1.25, bottom=1.2)

# 모든 기업에 대해 그래프를 그립니다.
for i, company in enumerate(company_list, 1):
    # 2x2의 그래프에서 i번째 위치에 그래프를 그립니다.
    plt.subplot(2, 2, i)
    
    # 기업의 종가 데이터를 그래프로 나타냅니다.
    company['Adj Close'].plot()
    
    # y축 라벨을 설정합니다.
    plt.ylabel('Adj Close')
    
    # x축 라벨은 생략합니다.
    plt.xlabel(None)
    
    # 그래프 제목을 설정합니다.
    plt.title(f"Closing Price of {tech_list[i - 1]}")
    
# 그래프들이 서로 겹치지 않도록 자동으로 간격을 조절합니다.
plt.tight_layout()


# In[5]:


# 그래프의 크기를 지정합니다.
plt.figure(figsize=(15, 10))

# 그래프 간의 간격을 조절합니다.
plt.subplots_adjust(top=1.25, bottom=1.2)

# 모든 기업에 대해 그래프를 그립니다.
for i, company in enumerate(company_list, 1):
    # 2x2의 그래프에서 i번째 위치에 그래프를 그립니다.
    plt.subplot(2, 2, i)
    
    # 기업의 거래량 데이터를 그래프로 나타냅니다.
    company['Volume'].plot()
    
    # y축 라벨을 설정합니다.
    plt.ylabel('Volume')
    
    # x축 라벨은 생략합니다.
    plt.xlabel(None)
    
    # 그래프 제목을 설정합니다.
    plt.title(f"Sales Volume for {tech_list[i - 1]}")
    
# 그래프들이 서로 겹치지 않도록 자동으로 간격을 조절합니다.
plt.tight_layout()


# In[6]:


# 이동평균을 계산할 날짜 리스트를 정의합니다.
ma_day = [10, 20, 50]

# 모든 기업에 대해 이동평균을 계산하고, 이동평균 값을 컬럼으로 추가합니다.
for ma in ma_day:
    for company in company_list:
        column_name = f"MA for {ma} days"
        company[column_name] = company['Adj Close'].rolling(ma).mean()

# 그래프를 그리기 위해 그래프 크기를 설정합니다.
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

# APPLE 주식에 대한 그래프를 그립니다.
AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,0])
axes[0,0].set_title('APPLE')

# GOOGLE 주식에 대한 그래프를 그립니다.
GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,1])
axes[0,1].set_title('GOOGLE')

# MICROSOFT 주식에 대한 그래프를 그립니다.
MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,0])
axes[1,0].set_title('MICROSOFT')

# AMAZON 주식에 대한 그래프를 그립니다.
AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,1])
axes[1,1].set_title('AMAZON')

# 서로 겹치지 않게 그래프를 조정합니다.
fig.tight_layout()


# In[7]:


# 모든 기업에 대해 일별 주가 변동율을 계산하고, Daily Return 컬럼으로 추가합니다.
for company in company_list:
    company['Daily Return'] = company['Adj Close'].pct_change()

# 그래프를 그리기 위해 그래프 크기를 설정합니다.
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

# APPLE 주식의 일별 주가 변동율 그래프를 그립니다.
AAPL['Daily Return'].plot(ax=axes[0,0], legend=True, linestyle='--', marker='o')
axes[0,0].set_title('APPLE')

# GOOGLE 주식의 일별 주가 변동율 그래프를 그립니다.
GOOG['Daily Return'].plot(ax=axes[0,1], legend=True, linestyle='--', marker='o')
axes[0,1].set_title('GOOGLE')

# MICROSOFT 주식의 일별 주가 변동율 그래프를 그립니다.
MSFT['Daily Return'].plot(ax=axes[1,0], legend=True, linestyle='--', marker='o')
axes[1,0].set_title('MICROSOFT')

# AMAZON 주식의 일별 주가 변동율 그래프를 그립니다.
AMZN['Daily Return'].plot(ax=axes[1,1], legend=True, linestyle='--', marker='o')
axes[1,1].set_title('AMAZON')

# 서로 겹치지 않게 그래프를 조정합니다.
fig.tight_layout()


# In[8]:


# 그래프의 크기를 지정합니다.
plt.figure(figsize=(12, 9))

# 모든 기업에 대해 일별 주가 변동율의 히스토그램을 그립니다.
for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Daily Return'].hist(bins=50)
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title(f'{company_name[i - 1]}')
    
# 그래프들이 서로 겹치지 않도록 자동으로 간격을 조절합니다.
plt.tight_layout()


# In[9]:


# tech_list에 저장된 기업들의 일별 수정 종가 데이터를 가져와서, 새로운 데이터프레임(closing_df)에 저장합니다.
closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']

# closing_df 데이터프레임의 일별 수익률을 계산하여, tech_rets라는 새로운 데이터프레임에 저장합니다.
tech_rets = closing_df.pct_change()

# tech_rets의 첫 5개 행을 출력합니다.
tech_rets.head()


# In[10]:


# seaborn 라이브러리를 사용하여, 구글(GOOG)의 일별 수익률 데이터를 산점도 그래프로 시각화합니다.
sns.jointplot(x='GOOG', y='GOOG', data=tech_rets, kind='scatter', color='seagreen')


# In[11]:


# We'll use joinplot to compare the daily returns of Google and Microsoft
sns.jointplot(x='GOOG', y='MSFT', data=tech_rets, kind='scatter')


# In[12]:


# seaborn 라이브러리를 사용하여, tech_rets 데이터프레임에 저장된 모든 기업들 간의 일별 수익률 비교를 위한 시각화를 자동으로 수행합니다.
# pairplot 함수를 호출하면 데이터프레임 내의 모든 수치형 변수 쌍에 대한 그래프를 그려줍니다.
# 각 그래프는 두 변수 간의 관계를 나타내며, 대각선에는 해당 변수의 분포를 보여주는 히스토그램이 그려집니다.
# kind='reg' 옵션을 사용하면 선형 회귀 직선이 함께 그려집니다.
sns.pairplot(tech_rets, kind='reg')


# In[13]:


# seaborn 라이브러리를 사용하여, 기업들 간의 일별 수익률을 비교하는 그래프를 조금 더 상세하게 그리는 코드입니다.

# PairGrid 함수를 사용하여 새로운 그래프를 만듭니다.
# tech_rets.dropna()를 사용하여 결측값이 있는 행을 제거합니다.
return_fig = sns.PairGrid(tech_rets.dropna())

# map_upper 함수를 사용하여 그래프 내부의 상단 삼각형에 대한 그래프 유형을 지정합니다.
# 여기서는 산점도 그래프를 사용하고, 색상은 보라색으로 지정합니다.
return_fig.map_upper(plt.scatter, color='purple')

# map_lower 함수를 사용하여 그래프 내부의 하단 삼각형에 대한 그래프 유형을 지정합니다.
# 여기서는 커널 밀도 추정(kdeplot) 그래프를 사용하고, 색상 맵은 cool_d로 지정합니다.
return_fig.map_lower(sns.kdeplot, cmap='cool_d')

# map_diag 함수를 사용하여 그래프 대각선에 대한 그래프 유형을 지정합니다.
# 여기서는 히스토그램을 사용하고, 막대의 개수(bins)는 30으로 지정합니다.
return_fig.map_diag(plt.hist, bins=30)


# In[14]:


# seaborn 라이브러리를 사용하여, 각 기업들의 일별 수정 종가를 비교하는 그래프를 그리는 코드입니다.

# PairGrid 함수를 사용하여 그래프를 만듭니다.
returns_fig = sns.PairGrid(closing_df)

# map_upper 함수를 사용하여 그래프 내부의 상단 삼각형에 대한 그래프 유형을 지정합니다.
# 여기서는 산점도 그래프를 사용하고, 색상은 보라색으로 지정합니다.
returns_fig.map_upper(plt.scatter,color='purple')

# map_lower 함수를 사용하여 그래프 내부의 하단 삼각형에 대한 그래프 유형을 지정합니다.
# 여기서는 커널 밀도 추정(kdeplot) 그래프를 사용하고, 색상 맵은 cool_d로 지정합니다.
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

# map_diag 함수를 사용하여 그래프 대각선에 대한 그래프 유형을 지정합니다.
# 여기서는 히스토그램을 사용하고, 막대의 개수(bins)는 30으로 지정합니다.
returns_fig.map_diag(plt.hist,bins=30)


# In[15]:


# seaborn 라이브러리를 사용하여, 각 기업들의 일별 수익률과 수정 종가 간의 상관 관계를 나타내는 히트맵을 그리는 코드입니다.

# subplot 함수를 사용하여 2x2 크기의 그래프를 만듭니다.
plt.figure(figsize=(12, 10))

# 첫 번째 그래프는 tech_rets DataFrame의 상관 관계를 나타내는 히트맵입니다.
plt.subplot(2, 2, 1)
sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
plt.title('Correlation of stock return')

# 두 번째 그래프는 closing_df DataFrame의 상관 관계를 나타내는 히트맵입니다.
plt.subplot(2, 2, 2)
sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
plt.title('Correlation of stock closing price')


# In[16]:


# numpy와 matplotlib 라이브러리를 사용하여, 각 기업들의 일별 수익률에 따른 위험과 예상 수익률을 나타내는 산점도 그래프를 그리는 코드입니다.

# tech_rets DataFrame에서 결측값을 제거한 후 rets 변수에 할당합니다.
rets = tech_rets.dropna()

# numpy의 pi 상수와 20을 곱하여 area 변수에 할당합니다.
area = np.pi * 20

# plt.scatter 함수를 사용하여 산점도 그래프를 그리는데,
# x축은 일별 수익률의 평균, y축은 일별 수익률의 표준편차를 나타내며, 점의 크기는 area 변수로 지정됩니다.
plt.figure(figsize=(10, 8))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')

# annotate 함수를 사용하여 각 점에 대한 기업 이름을 표시합니다.
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                 arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))


# In[17]:


# pandas_datareader 라이브러리를 사용하여, 2012년 1월 1일부터 현재까지의 AAPL(애플) 주식 데이터를 불러와서 DataFrame으로 표시하는 코드입니다.

# 먼저, pdr.get_data_yahoo 함수를 사용하여 애플의 주식 데이터를 불러오고,
# start와 end 인자를 사용하여 불러올 데이터의 기간을 설정합니다.
df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())

# 마지막으로, df 변수에 불러온 데이터를 할당합니다.
# df를 출력하여 데이터를 확인합니다.
df


# In[18]:


# matplotlib 라이브러리를 사용하여, 애플 주식의 종가(close price) 추이를 그래프로 나타내는 코드입니다.

# plt.figure 함수를 사용하여 그래프의 크기를 설정하고,
# plt.title 함수를 사용하여 그래프의 제목을 설정합니다.
plt.figure(figsize=(16,6))
plt.title('Close Price History')

# plt.plot 함수를 사용하여 애플 주식의 종가 데이터(df['Close'])를 그래프로 나타냅니다.
# x축에는 날짜(Date)를, y축에는 종가(Close Price USD ($))를 나타냅니다.
plt.plot(df['Close'])

# plt.xlabel, plt.ylabel 함수를 사용하여 x축과 y축의 레이블을 설정합니다.
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)

# plt.show 함수를 사용하여 그래프를 출력합니다.
plt.show()


# In[19]:


# pandas 라이브러리를 사용하여, 애플 주식의 종가 데이터를 numpy 배열로 변환하고,
# 학습 데이터(training data)의 크기를 계산하는 코드입니다.

# df.filter 함수를 사용하여 'Close' 열(column)만을 포함한 새로운 데이터프레임(data)을 만듭니다.
data = df.filter(['Close'])

# data.values를 사용하여 데이터프레임을 numpy 배열(dataset)로 변환합니다.
dataset = data.values

# int(np.ceil( len(dataset) * .95 ))를 사용하여 학습 데이터(training data)의 크기를 계산합니다.
# 학습 데이터의 크기는 전체 데이터셋의 95%로 설정되며,
# np.ceil 함수를 사용하여 소수점 이하의 값을 올림(round up) 처리합니다.
training_data_len = int(np.ceil( len(dataset) * .95 ))

# training_data_len을 출력하여 학습 데이터의 크기를 확인합니다.
training_data_len


# In[20]:


# scikit-learn 라이브러리를 사용하여, numpy 배열로 변환된 애플 주식의 종가 데이터를 스케일링하는 코드입니다.

# from sklearn.preprocessing import MinMaxScaler를 사용하여 MinMaxScaler 클래스를 임포트합니다.
from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler 객체(scaler)를 생성합니다.
scaler = MinMaxScaler(feature_range=(0,1))

# scaler.fit_transform 함수를 사용하여 scaled_data에 스케일링된 데이터를 저장합니다.
scaled_data = scaler.fit_transform(dataset)

# scaled_data를 출력하여 스케일링된 데이터를 확인합니다.
scaled_data


# In[21]:


# 스케일링된 애플 주식 종가 데이터를 학습 데이터셋으로 변환하는 코드입니다.

# train_data에 95% 학습 데이터(training data)를 저장합니다.
train_data = scaled_data[0:int(training_data_len), :]

# train_data를 60일씩 슬라이싱하여 x_train과 y_train 데이터셋으로 분할합니다.
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()

# x_train과 y_train을 numpy 배열로 변환합니다.
x_train, y_train = np.array(x_train), np.array(y_train)

# x_train의 크기를 (x_train.shape[0], x_train.shape[1], 1)로 변환하여 3차원으로 만들어 줍니다.
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# x_train의 크기를 출력하여 데이터셋의 크기를 확인합니다.
x_train.shape


# In[22]:


# LSTM(Long Short-Term Memory) 모델을 구성하고 학습시키는 코드입니다.

# keras.models 패키지에서 Sequential 클래스와 layers 패키지에서 Dense와 LSTM 클래스를 import합니다.
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Sequential() 함수를 사용하여 모델 객체 model을 생성합니다.
model = Sequential()

# model.add() 함수를 사용하여 LSTM 층을 추가합니다.
# 첫 번째 LSTM 층은 128개의 뉴런(neuron)을 가지며, return_sequences=True로 설정하여 다음 층에서도 시퀀스(Sequence) 출력을 유지하도록 합니다.
# input_shape은 x_train 데이터셋의 차원을 지정합니다.
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))

# 두 번째 LSTM 층은 64개의 뉴런을 가지며, return_sequences=False로 설정하여 마지막 층에서 시퀀스 출력이 필요하지 않도록 합니다.
model.add(LSTM(64, return_sequences=False))

# Dense() 함수를 사용하여 완전 연결층(fully connected layer)을 추가합니다.
model.add(Dense(25))
model.add(Dense(1))

# model.compile() 함수를 사용하여 모델을 컴파일합니다. optimizer로 adam을, loss로 mean squared error를 설정합니다.
model.compile(optimizer='adam', loss='mean_squared_error')

# model.fit() 함수를 사용하여 모델을 학습합니다. x_train과 y_train 데이터셋을 입력으로 하여 batch size를 1, epochs를 1로 설정합니다.
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[23]:


# 테스트 데이터 셋 만들기
# 새로운 배열을 생성하고 스케일링된 값을 인덱스 1543에서 2002까지 저장
test_data = scaled_data[training_data_len - 60: , :]
# 데이터 셋 x_test와 y_test 생성
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    # x_test 배열에 60일치의 데이터를 넣음
    x_test.append(test_data[i-60:i, 0])

# 데이터를 numpy 배열로 변환
x_test = np.array(x_test)

# 데이터 형태 재조정
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# 모델로 예측한 가격 가져오기
predictions = model.predict(x_test)

print(x_test)
print(len(x_test))
print(predictions)
print(len(predictions))
# 스케일링 된 값을 되돌리기
predictions = scaler.inverse_transform(predictions)

# 평균제곱근오차 (RMSE) 계산하기
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse


# In[24]:


# 예측한 값을 시각화해서 확인해보자
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# 데이터 시각화
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[25]:


# Show the valid and predicted prices
valid


# In[76]:


# 주식 정보 가져오기
df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())

# 'Close' 열만 포함하는 새로운 데이터프레임 생성
data = df.filter(['Close'])

# 최근 60일 종가 가져오기
last_60_days = data[-60:].values

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0,1))
last_60_days_scaled = scaler.fit_transform(last_60_days)


# In[80]:


# 예측값 저장할 빈 리스트 생성
prediction_list = []

# 다음 5일 예측값을 생성하고 리스트에 추가

for i in range(10):
    # 현재 마지막 60일치 데이터를 모델에 입력할 수 있는 형태로 변형
    x_test = np.reshape(last_60_days_scaled, (1, last_60_days_scaled.shape[0], 1))
    # 모델로부터 종가 예측값 계산
    prediction_scaled = model.predict(x_test)
    # 예측값 역 스케일링
    prediction = scaler.inverse_transform(prediction_scaled)
    # 리스트에 예측값 추가
    prediction_list.append(prediction[0][0])

    # 마지막 60일치 데이터에 예측값을 추가하여 다음 예측에 사용
    last_60_days_scaled = np.append(last_60_days_scaled, prediction_scaled, axis=0)

    # 마지막 60일치 데이터에서 첫 번째 데이터 삭제하여 다음 예측에 사용
    last_60_days_scaled = np.delete(last_60_days_scaled, 0, axis=0)

# 다음 5일의 날짜 리스트 생성
last_day = df.index[-1]
date_list = pd.date_range(last_day, periods=10).tolist()

# 다음 10일 예측값과 해당하는 날짜 출력
for i in range(10):
    print(f"Date: {date_list[i].date()} | Predicted Close Price: ${prediction_list[i]:.2f}")


# In[78]:


# Visualize the predicted values
plt.figure(figsize=(16,6))
plt.plot(data.index, data['Close'])
plt.plot(date_list, prediction_list)
plt.title('AAPL Close Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(['Historical', 'Predicted'])
plt.show()


# In[ ]:




