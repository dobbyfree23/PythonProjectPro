from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
boston = fetch_openml(name='boston')

# 입력 데이터와 타깃 데이터 분리
X = boston.data
y = boston.target

# 데이터 분리: 학습 데이터와 검증 데이터
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 검증 데이터를 사용하여 예측 결과 계산
y_pred = model.predict(X_test)

# 평균 제곱 오차 계산
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)