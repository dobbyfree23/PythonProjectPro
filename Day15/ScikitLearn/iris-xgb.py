import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 붓꽃(iris) 데이터셋 로드
iris = load_iris()

# 데이터셋 분리
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 모델 학습
params = {
    'objective': 'multi:softmax',   # 목적 함수: 다중 클래스 분류 문제
    'num_class': 3,                 # 클래스 개수: 3
    'max_depth': 3,                 # 의사결정나무의 최대 깊이: 3
    'eta': 0.1,                     # 학습률: 0.1
    'n_estimators': 100             # 의사결정나무의 개수: 100
}

# DMatrix 객체 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 모델 학습
model = xgb.train(params, dtrain)

# 모델 예측
y_pred = model.predict(dtest)

# 정확도 출력
print("Accuracy:", accuracy_score(y_test, y_pred))