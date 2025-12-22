# Iris Species Prediction - End-to-End ML Project

A complete machine learning project with Streamlit web app, model training, and CI/CD pipeline for predicting iris species based on flower measurements.

**Last Updated**: December 22, 2025 ✅

## 🌸 Features

- **Machine Learning Model**: Trained SVM classifier with hyperparameter tuning
- **Streamlit Web App**: Interactive user interface for making predictions
- **Data Analysis**: Comprehensive exploratory data analysis notebook
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Model Persistence**: Saved trained models and preprocessing objects

## 📊 Project Structure

```
iris_ci_cd/
├── iris_analysis.ipynb      # Comprehensive EDA and model development
├── train_model.py           # Model training and saving script
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore file
├── models/                 # Trained models directory
│   ├── iris_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
└── .github/workflows/
    └── ci-cd.yml           # GitHub Actions workflow
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📈 Model Performance

- **Algorithm**: Support Vector Machine (SVM)
- **Test Accuracy**: ~93.33%
- **Cross-Validation Score**: ~97.50%
- **Classes**: Setosa, Versicolor, Virginica

### Hyperparameters
- **Kernel**: Linear
- **C**: 0.1
- **Gamma**: Scale

## 🎯 Features Input

The model accepts the following iris measurements:

1. **Sepal Length** (4.0 - 8.0 cm)
2. **Sepal Width** (2.0 - 4.5 cm)
3. **Petal Length** (1.0 - 7.0 cm)
4. **Petal Width** (0.1 - 2.5 cm)

## 📱 Streamlit App Features

### Prediction Tab
- Interactive sliders for feature input
- Real-time predictions with confidence scores
- Probability distribution visualization
- Input features summary

### Model Info Tab
- Model type and configuration
- Feature and class information
- Model training details

### Example Data Tab
- Pre-loaded example predictions
- Typical measurements for each iris species

## 🔄 CI/CD Pipeline

The project includes a GitHub Actions workflow that:

- ✅ Runs automated tests on every push
- ✅ Validates model performance
- ✅ Checks code quality
- ✅ Builds and deploys the app

### Workflow Triggers
- Push to main branch
- Pull requests

## 📦 Model Deployment

The trained model is saved with preprocessing objects:

- `iris_model.pkl` - Trained SVM classifier
- `scaler.pkl` - StandardScaler for feature normalization
- `label_encoder.pkl` - Label encoder for target classes
- `feature_names.pkl` - Feature names for validation

## 🧪 Testing

Run tests with:
```bash
pytest
```

## 📚 Dataset

The project uses the classic Iris dataset:
- **Source**: sklearn.datasets
- **Samples**: 150
- **Features**: 4
- **Classes**: 3 (Setosa, Versicolor, Virginica)

## 🛠️ Development

### Adding New Features

1. Update feature inputs in `app.py`
2. Retrain the model: `python train_model.py`
3. Test the predictions
4. Commit and push changes

### Improving Model Performance

1. Modify hyperparameters in `train_model.py`
2. Train and evaluate: `python train_model.py`
3. Update test thresholds if needed
4. Commit changes

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Iris Species Prediction Project - December 2025

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)
- [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
