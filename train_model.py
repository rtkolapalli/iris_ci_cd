"""
Train and save the Iris classification model
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os

def train_and_save_model():
    """Train the model and save it along with preprocessing objects"""
    
    # Load the iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target, columns=['species'])
    y['species'] = y['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # Combine features and target
    df = pd.concat([X, y], axis=1)
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['species'])
    
    # Separate features and target
    X = df.iloc[:, :-1]
    y = y_encoded
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Hyperparameter tuning for SVM
    print("Training and tuning SVM model...")
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    
    grid_search = GridSearchCV(
        SVC(random_state=42, probability=True),
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Evaluate
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    print(f"\nModel Training Complete!")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Train Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and preprocessing objects
    with open('models/iris_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    print("\nModel and preprocessing objects saved successfully!")

if __name__ == '__main__':
    train_and_save_model()
