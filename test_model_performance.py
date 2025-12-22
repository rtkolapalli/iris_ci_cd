"""Test model performance"""
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

iris = load_iris()
X = iris.data
y = iris.target
le = LabelEncoder()
y = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
model = pickle.load(open('models/iris_model.pkl', 'rb'))
accuracy = model.score(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')
if accuracy >= 0.85:
    print('✅ Model performance verified!')
else:
    print(f'⚠️ Accuracy {accuracy:.4f} below 85% threshold')
