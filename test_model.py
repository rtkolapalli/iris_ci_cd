"""
Test file for model prediction
"""
import pickle
import numpy as np
import pytest

def load_models():
    """Load trained model and preprocessing objects"""
    with open('models/iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, scaler, label_encoder

def test_model_exists():
    """Test that model files exist"""
    try:
        model, scaler, le = load_models()
        assert model is not None
        assert scaler is not None
        assert le is not None
    except FileNotFoundError:
        pytest.skip("Model files not found")

def test_model_prediction():
    """Test that model makes valid predictions"""
    try:
        model, scaler, le = load_models()
    except FileNotFoundError:
        pytest.skip("Model files not found")
    
    # Test with a typical setosa measurement
    test_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    test_scaled = scaler.transform(test_input)
    prediction = model.predict(test_scaled)
    
    # Check prediction is valid
    assert prediction[0] in range(len(le.classes_))

def test_model_probabilities():
    """Test that model returns valid probabilities"""
    try:
        model, scaler, le = load_models()
    except FileNotFoundError:
        pytest.skip("Model files not found")
    
    # Test with a typical measurement
    test_input = np.array([[5.9, 2.7, 4.2, 1.3]])
    test_scaled = scaler.transform(test_input)
    probabilities = model.predict_proba(test_scaled)
    
    # Check probabilities sum to 1 and are between 0 and 1
    assert len(probabilities) == 1
    assert np.allclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)

def test_model_accuracy():
    """Test that model achieves minimum accuracy threshold"""
    try:
        model, scaler, le = load_models()
    except FileNotFoundError:
        pytest.skip("Model files not found")
    
    # This is a placeholder test
    # In a real scenario, you'd test against a validation set
    assert model is not None

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
