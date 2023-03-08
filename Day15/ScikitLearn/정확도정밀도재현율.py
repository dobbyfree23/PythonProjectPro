from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 데이터셋 로드
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# train/test 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 생성 및 학습
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# 예측 결과 출력
y_pred = clf.predict(X_test)

# 정확도, 정밀도, 재현율 출력
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))