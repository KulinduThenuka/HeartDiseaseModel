import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

class HeartDiseasePredictor:
    """
    Heart Disease Prediction using saved model
    """
    
    def __init__(self, model_path, scaler_path, poly_path, feature_names_path):
        """
        Initialize the predictor with saved model components
        
        Parameters:
        - model_path: path to saved logistic regression model
        - scaler_path: path to saved StandardScaler
        - poly_path: path to saved PolynomialFeatures transformer
        - feature_names_path: path to saved feature names
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.poly_path = poly_path
        self.feature_names_path = feature_names_path
        
        # Load all components
        self.load_model_components()
        
    def load_model_components(self):
        """Load all saved model components"""
        try:
            print("🔄 Loading model components...")
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.poly = joblib.load(self.poly_path)
            self.feature_names = joblib.load(self.feature_names_path)
            print("✅ All components loaded successfully!")
            print(f"📋 Required features: {self.feature_names}")
        except Exception as e:
            print(f"❌ Error loading model components: {e}")
            raise
    
    def predict_single_patient(self, patient_data):
        """
        Predict heart disease for a single patient
        
        Parameters:
        - patient_data: dictionary with patient features
        
        Returns:
        - prediction: 0 (no disease) or 1 (disease)
        - probability: probability of having heart disease
        - risk_level: risk assessment string
        """
        try:
            # Convert to DataFrame
            if isinstance(patient_data, dict):
                input_df = pd.DataFrame([patient_data])
            else:
                input_df = patient_data.copy()
            
            # Validate features
            missing_features = [f for f in self.feature_names if f not in input_df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Select and order features correctly
            input_df = input_df[self.feature_names]
            
            # Apply preprocessing pipeline
            input_scaled = self.scaler.transform(input_df)
            input_poly = self.poly.transform(input_scaled)
            
            # Make prediction
            prediction = self.model.predict(input_poly)[0]
            probability = self.model.predict_proba(input_poly)[0, 1]
            
            # Risk assessment
            if probability >= 0.8:
                risk_level = "🔴 VERY HIGH RISK"
                recommendation = "Immediate medical attention required!"
            elif probability >= 0.6:
                risk_level = "🟠 HIGH RISK"
                recommendation = "Consult a cardiologist soon"
            elif probability >= 0.4:
                risk_level = "🟡 MODERATE RISK"
                recommendation = "Regular health monitoring advised"
            else:
                risk_level = "🟢 LOW RISK"
                recommendation = "Continue healthy lifestyle"
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level,
                'recommendation': recommendation
            }
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def predict_multiple_patients(self, patients_df):
        """
        Predict heart disease for multiple patients
        
        Parameters:
        - patients_df: DataFrame with multiple patient records
        
        Returns:
        - DataFrame with predictions and probabilities
        """
        try:
            # Validate features
            missing_features = [f for f in self.feature_names if f not in patients_df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Select and order features correctly
            input_df = patients_df[self.feature_names].copy()
            
            # Apply preprocessing pipeline
            input_scaled = self.scaler.transform(input_df)
            input_poly = self.poly.transform(input_scaled)
            
            # Make predictions
            predictions = self.model.predict(input_poly)
            probabilities = self.model.predict_proba(input_poly)[:, 1]
            
            # Create results DataFrame
            results_df = patients_df.copy()
            results_df['prediction'] = predictions
            results_df['probability'] = probabilities
            results_df['predicted_class'] = results_df['prediction'].map({0: 'No Disease', 1: 'Heart Disease'})
            
            # Add risk levels
            def get_risk_level(prob):
                if prob >= 0.8:
                    return "🔴 VERY HIGH RISK"
                elif prob >= 0.6:
                    return "🟠 HIGH RISK"
                elif prob >= 0.4:
                    return "🟡 MODERATE RISK"
                else:
                    return "🟢 LOW RISK"
            
            results_df['risk_level'] = results_df['probability'].apply(get_risk_level)
            
            return results_df
            
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return None

    def print_feature_info(self):
        """Print information about required features"""
        feature_info = {
            'age': 'Age in years',
            'sex': 'Sex (1 = male; 0 = female)',
            'cp': 'Chest pain type (0-3)',
            'trestbps': 'Resting blood pressure (in mm Hg)',
            'chol': 'Serum cholesterol in mg/dl',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
            'restecg': 'Resting electrocardiographic results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1 = yes; 0 = no)',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'slope': 'The slope of the peak exercise ST segment (0-2)',
            'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
            'thal': 'Thalassemia (0-3)'
        }
        
        print("\n📋 REQUIRED FEATURES AND DESCRIPTIONS:")
        print("=" * 60)
        for feature in self.feature_names:
            print(f"{feature:12}: {feature_info.get(feature, 'Feature description')}")
        print("=" * 60)

# =============================================================================
# MAIN PREDICTION SCRIPT
# =============================================================================

# def main():
#     """Main function to demonstrate model usage"""
    
#     print("🏥 HEART DISEASE PREDICTION SYSTEM")
#     print("=" * 50)
    
#     # 1. SET YOUR MODEL PATHS HERE
#     # Replace these with your actual saved model paths
#     model_path = "models/heart_disease_model_20250803_191519.pkl" 
#     scaler_path = "scalers/scaler_20250803_191519.pkl"           
#     poly_path = "scalers/poly_transformer_20250803_191519.pkl"    
#     feature_names_path = "scalers/feature_names_20250803_191519.pkl"  
    
#     # Check if files exist
#     paths = [model_path, scaler_path, poly_path, feature_names_path]
#     missing_files = [path for path in paths if not os.path.exists(path)]
    
#     if missing_files:
#         print("❌ Missing model files:")
#         for file in missing_files:
#             print(f"   - {file}")
#         print("\n💡 Please update the file paths in the script or ensure files exist.")
        
#         # Show available model files
#         if os.path.exists('models'):
#             print("\n📁 Available model files in 'models/' directory:")
#             for file in os.listdir('models'):
#                 if file.endswith('.pkl'):
#                     print(f"   - {file}")
#         return
    
#     # 2. Initialize predictor
#     try:
#         predictor = HeartDiseasePredictor(model_path, scaler_path, poly_path, feature_names_path)
#     except Exception as e:
#         print(f"❌ Failed to initialize predictor: {e}")
#         return
    
#     # 3. Show feature information
#     predictor.print_feature_info()
    
#     # 4. EXAMPLE 1: Single Patient Prediction
#     print("\n🧑‍⚕️ EXAMPLE 1: SINGLE PATIENT PREDICTION")
#     print("-" * 50)
    
#     # Example patient data (modify these values as needed)
#     patient_1 = {
#         'age': 63,
#         'sex': 1,        # Male
#         'cp': 3,         # Chest pain type
#         'trestbps': 145, # Resting blood pressure
#         'chol': 233,     # Cholesterol
#         'fbs': 1,        # Fasting blood sugar > 120
#         'restecg': 0,    # Resting ECG
#         'thalach': 150,  # Max heart rate
#         'exang': 0,      # Exercise induced angina
#         'oldpeak': 2.3,  # ST depression
#         'slope': 0,      # Slope of peak exercise ST
#         'ca': 0,         # Number of major vessels
#         'thal': 1        # Thalassemia
#     }
    
#     print("👤 Patient Information:")
#     for key, value in patient_1.items():
#         print(f"   {key}: {value}")
    
#     # Make prediction
#     result = predictor.predict_single_patient(patient_1)
    
#     if result:
#         print(f"\n📊 PREDICTION RESULTS:")
#         print(f"   Prediction: {'❤️ HEART DISEASE' if result['prediction'] == 1 else '💚 NO HEART DISEASE'}")
#         print(f"   Probability: {result['probability']:.3f} ({result['probability']*100:.1f}%)")
#         print(f"   Risk Level: {result['risk_level']}")
#         print(f"   Recommendation: {result['recommendation']}")
    
#     # 5. EXAMPLE 2: Multiple Patients Prediction
#     print("\n\n👥 EXAMPLE 2: MULTIPLE PATIENTS PREDICTION")
#     print("-" * 50)
    
#     # Example multiple patients data
#     patients_data = [
#         {
#             'age': 67, 'sex': 1, 'cp': 0, 'trestbps': 160, 'chol': 286,
#             'fbs': 0, 'restecg': 0, 'thalach': 108, 'exang': 1, 'oldpeak': 1.5,
#             'slope': 1, 'ca': 3, 'thal': 2
#         },
#         {
#             'age': 37, 'sex': 1, 'cp': 2, 'trestbps': 130, 'chol': 250,
#             'fbs': 0, 'restecg': 1, 'thalach': 187, 'exang': 0, 'oldpeak': 3.5,
#             'slope': 0, 'ca': 0, 'thal': 2
#         },
#         {
#             'age': 41, 'sex': 0, 'cp': 1, 'trestbps': 130, 'chol': 204,
#             'fbs': 0, 'restecg': 0, 'thalach': 172, 'exang': 0, 'oldpeak': 1.4,
#             'slope': 2, 'ca': 0, 'thal': 2
#         }
#     ]
    
#     patients_df = pd.DataFrame(patients_data)
#     print(f"📊 Predicting for {len(patients_df)} patients...")
    
#     # Make predictions
#     results_df = predictor.predict_multiple_patients(patients_df)
    
#     if results_df is not None:
#         print("\n📋 RESULTS SUMMARY:")
#         for idx, row in results_df.iterrows():
#             print(f"\n   Patient {idx + 1}:")
#             print(f"      Age: {row['age']}, Sex: {'Male' if row['sex'] == 1 else 'Female'}")
#             print(f"      Prediction: {row['predicted_class']}")
#             print(f"      Probability: {row['probability']:.3f} ({row['probability']*100:.1f}%)")
#             print(f"      Risk Level: {row['risk_level']}")
        
#         # Save results to CSV
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_file = f"prediction_results_{timestamp}.csv"
#         results_df.to_csv(output_file, index=False)
#         print(f"\n💾 Results saved to: {output_file}")

#     print("\n🎉 Prediction completed successfully!")

# =============================================================================
# INTERACTIVE PREDICTION FUNCTION
# =============================================================================

def interactive_prediction():
    """Interactive function to input patient data and get predictions"""
    
    print("\n🔮 INTERACTIVE HEART DISEASE PREDICTION")
    print("=" * 50)
    
    # Update these paths with your actual model paths
    model_path = "models/heart_disease_model_20250803_191519.pkl" 
    scaler_path = "scalers/scaler_20250803_191519.pkl"           
    poly_path = "scalers/poly_transformer_20250803_191519.pkl"    
    feature_names_path = "scalers/feature_names_20250803_191519.pkl"  
    
    try:
        predictor = HeartDiseasePredictor(model_path, scaler_path, poly_path, feature_names_path)
    except:
        print("❌ Could not load model. Please check file paths.")
        return
    
    print("\n📝 Please enter patient information:")
    print("(Press Enter to use default values shown in brackets)")
    
    patient_data = {}
    
    # Input fields with defaults and validation
    inputs = {
        'age': ('Age in years', 50, lambda x: 1 <= x <= 120),
        'sex': ('Sex (0=Female, 1=Male)', 1, lambda x: x in [0, 1]),
        'cp': ('Chest pain type (0-3)', 0, lambda x: 0 <= x <= 3),
        'trestbps': ('Resting blood pressure (mm Hg)', 120, lambda x: 50 <= x <= 300),
        'chol': ('Serum cholesterol (mg/dl)', 200, lambda x: 100 <= x <= 600),
        'fbs': ('Fasting blood sugar > 120 mg/dl (0=No, 1=Yes)', 0, lambda x: x in [0, 1]),
        'restecg': ('Resting ECG results (0-2)', 0, lambda x: 0 <= x <= 2),
        'thalach': ('Maximum heart rate achieved', 150, lambda x: 50 <= x <= 250),
        'exang': ('Exercise induced angina (0=No, 1=Yes)', 0, lambda x: x in [0, 1]),
        'oldpeak': ('ST depression induced by exercise', 0.0, lambda x: 0 <= x <= 10),
        'slope': ('Slope of peak exercise ST segment (0-2)', 0, lambda x: 0 <= x <= 2),
        'ca': ('Number of major vessels (0-3)', 0, lambda x: 0 <= x <= 3),
        'thal': ('Thalassemia (0-3)', 1, lambda x: 0 <= x <= 3)
    }
    
    for field, (description, default, validator) in inputs.items():
        while True:
            try:
                user_input = input(f"{description} [{default}]: ").strip()
                if user_input == "":
                    value = default
                else:
                    value = float(user_input) if field == 'oldpeak' else int(user_input)
                
                if validator(value):
                    patient_data[field] = value
                    break
                else:
                    print(f"   ⚠️  Invalid value. Please try again.")
            except ValueError:
                print(f"   ⚠️  Please enter a valid number.")
    
    # Make prediction
    print("\n🔄 Making prediction...")
    result = predictor.predict_single_patient(patient_data)
    
    if result:
        print(f"\n📊 PREDICTION RESULTS:")
        print("=" * 30)
        print(f"Prediction: {'❤️ HEART DISEASE DETECTED' if result['prediction'] == 1 else '💚 NO HEART DISEASE'}")
        print(f"Probability: {result['probability']:.3f} ({result['probability']*100:.1f}%)")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        print("=" * 30)
        
        # Save individual prediction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"individual_prediction_{timestamp}.txt"
        
        # Convert risk level to text without emojis for file writing
        risk_level_text = result['risk_level'].replace('🔴', 'VERY HIGH RISK').replace('🟠', 'HIGH RISK').replace('🟡', 'MODERATE RISK').replace('🟢', 'LOW RISK')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Heart Disease Prediction Result\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Patient Data:\n")
            for key, value in patient_data.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nResults:\n")
            f.write(f"  Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}\n")
            f.write(f"  Probability: {result['probability']:.3f}\n")
            f.write(f"  Risk Level: {risk_level_text}\n")
            f.write(f"  Recommendation: {result['recommendation']}\n")
        
        print(f"💾 Results saved to: {filename}")

if __name__ == "__main__":
    
    # Uncomment the line below for interactive prediction
    interactive_prediction()