"""Test that model loads successfully"""
import pickle

try:
    model = pickle.load(open('models/iris_model.pkl', 'rb'))
    print('✅ Model loaded successfully')
    print(f'   Model type: {type(model).__name__}')
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
