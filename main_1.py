# Heart Disease Prediction Model - Complete Pipeline
# Includes: Training, Saving, Visualization, and Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib
import os
from datetime import datetime

# Create directories for saving
os.makedirs('models', exist_ok=True)
os.makedirs('graphs', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

print("🚀 Starting Heart Disease Prediction Model Pipeline....")

# 1. Load dataset
df = pd.read_csv('heart.csv')
print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Features and target
X = df.drop('target', axis=1)
y = df['target']

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Create polynomial interaction features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

print(f"🔧 Features after polynomial transformation: {X_poly.shape[1]} features")

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 6. GridSearchCV to tune regularization (C)
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000), param_grid, cv=5)
grid.fit(X_train, y_train)

# 7. Get best model and predictions
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("✅ Model Training Complete!")
print(f"✅ Best C value: {grid.best_params_}")
print(f"✅ Accuracy: {accuracy:.4f}")

# 9. SAVE THE MODEL AND PREPROCESSING COMPONENTS
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save the trained model
model_path = f'models/heart_disease_model_{timestamp}.pkl'
joblib.dump(best_model, model_path)

# Save the scaler
scaler_path = f'scalers/scaler_{timestamp}.pkl'
joblib.dump(scaler, scaler_path)

# Save the polynomial transformer
poly_path = f'scalers/poly_transformer_{timestamp}.pkl'
joblib.dump(poly, poly_path)

# Save feature names for reference
feature_names_path = f'scalers/feature_names_{timestamp}.pkl'
joblib.dump(X.columns.tolist(), feature_names_path)

print(f"💾 Model saved to: {model_path}")
print(f"💾 Scaler saved to: {scaler_path}")
print(f"💾 Poly transformer saved to: {poly_path}")
print(f"💾 Feature names saved to: {feature_names_path}")

# 10. CREATE VISUALIZATIONS

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Graph 1: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], 
            yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix - Heart Disease Prediction')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'graphs/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# Graph 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Heart Disease Prediction')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'graphs/roc_curve_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# Graph 3: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Heart Disease Prediction')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'graphs/precision_recall_curve_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# Graph 4: Feature Importance (Top 15)
feature_importance = abs(best_model.coef_[0])
poly_feature_names = poly.get_feature_names_out(X.columns)
importance_df = pd.DataFrame({
    'feature': poly_feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
plt.title('Top 15 Feature Importances (Logistic Regression Coefficients)')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.savefig(f'graphs/feature_importance_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# Graph 5: Model Performance Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
from sklearn.metrics import precision_score, recall_score, f1_score

values = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred)
]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.ylim(0, 1)
plt.title('Model Performance Metrics')
plt.ylabel('Score')

# Add value labels on bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'graphs/performance_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# Graph 6: Class Distribution
plt.figure(figsize=(8, 6))
class_counts = y.value_counts()
plt.pie(class_counts.values, labels=['No Disease', 'Disease'], autopct='%1.1f%%', 
        colors=['lightblue', 'lightcoral'], startangle=90)
plt.title('Heart Disease Class Distribution in Dataset')
plt.axis('equal')
plt.tight_layout()
plt.savefig(f'graphs/class_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 All graphs saved successfully!")

# 11. PREDICTION FUNCTION USING SAVED MODEL
def predict_heart_disease(model_path, scaler_path, poly_path, feature_names_path, input_data):
    """
    Function to make predictions using saved model
    
    Parameters:
    - model_path: path to saved model
    - scaler_path: path to saved scaler
    - poly_path: path to saved polynomial transformer
    - feature_names_path: path to saved feature names
    - input_data: dictionary with feature values or pandas DataFrame
    
    Returns:
    - prediction: 0 (no disease) or 1 (disease)
    - probability: probability of having heart disease
    """
    
    # Load saved components
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    poly = joblib.load(poly_path)
    feature_names = joblib.load(feature_names_path)
    
    # Convert input to DataFrame if it's a dictionary
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in input_df.columns:
            raise ValueError(f"Missing feature: {feature}")
    
    # Select and order features correctly
    input_df = input_df[feature_names]
    
    # Apply same preprocessing
    input_scaled = scaler.transform(input_df)
    input_poly = poly.transform(input_scaled)
    
    # Make prediction
    prediction = model.predict(input_poly)[0]
    probability = model.predict_proba(input_poly)[0, 1]
    
    return prediction, probability

# 12. EXAMPLE USAGE OF PREDICTION FUNCTION
print("\n" + "="*50)
print("🔮 MAKING PREDICTIONS WITH SAVED MODEL")
print("="*50)

# Example patient data (you can modify these values)
example_patient = {
    'age': 63,
    'sex': 1,  # 1 = male, 0 = female
    'cp': 3,   # chest pain type
    'trestbps': 145,  # resting blood pressure
    'chol': 233,      # cholesterol
    'fbs': 1,         # fasting blood sugar > 120 mg/dl
    'restecg': 0,     # resting ECG results
    'thalach': 150,   # max heart rate achieved
    'exang': 0,       # exercise induced angina
    'oldpeak': 2.3,   # ST depression
    'slope': 0,       # slope of peak exercise ST segment
    'ca': 0,          # number of major vessels colored by fluoroscopy
    'thal': 1         # thalassemia
}

# Make prediction
try:
    pred, prob = predict_heart_disease(model_path, scaler_path, poly_path, 
                                     feature_names_path, example_patient)
    
    print(f"🏥 Patient Information:")
    for key, value in example_patient.items():
        print(f"   {key}: {value}")
    
    print(f"\n📋 Prediction Results:")
    print(f"   Prediction: {'❤️ HEART DISEASE DETECTED' if pred == 1 else '💚 NO HEART DISEASE'}")
    print(f"   Probability of Heart Disease: {prob:.3f} ({prob*100:.1f}%)")
    
    if prob > 0.7:
        print("   ⚠️  HIGH RISK - Immediate medical consultation recommended")
    elif prob > 0.5:
        print("   ⚡ MODERATE RISK - Regular monitoring advised")
    else:
        print("   ✅ LOW RISK - Continue healthy lifestyle")
        
except Exception as e:
    print(f"❌ Error making prediction: {e}")

# 13. SAVE MODEL PATHS TO A TEXT FILE
paths_info = f"""
HEART DISEASE PREDICTION MODEL - SAVED PATHS
============================================
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Model Path: {model_path}
Scaler Path: {scaler_path}
Polynomial Transformer Path: {poly_path}
Feature Names Path: {feature_names_path}

Model Performance:
- Accuracy: {accuracy:.4f}
- Best C Parameter: {grid.best_params_['C']}
- ROC AUC: {roc_auc:.4f}

Usage Example:
prediction, probability = predict_heart_disease(
    '{model_path}',
    '{scaler_path}', 
    '{poly_path}',
    '{feature_names_path}',
    patient_data_dict
)

Graphs saved in 'graphs/' directory:
- confusion_matrix_{timestamp}.png
- roc_curve_{timestamp}.png
- precision_recall_curve_{timestamp}.png
- feature_importance_{timestamp}.png
- performance_metrics_{timestamp}.png
- class_distribution_{timestamp}.png
"""

with open(f'models/model_info_{timestamp}.txt', 'w') as f:
    f.write(paths_info)

print(f"\n📄 Model information saved to: models/model_info_{timestamp}.txt")
print("\n🎉 Pipeline completed successfully!")
print(f"📁 Check 'models/', 'scalers/', and 'graphs/' directories for all saved files.")
