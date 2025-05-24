from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from joblib import load, dump
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load or train the SVM model
def train_model():
    try:
        # Load existing model and scaler
        model = load('svm_model.joblib')
        scaler = load('scaler.joblib')
        logger.info("Loaded existing model and scaler.")
        return model, scaler
    except FileNotFoundError:
        logger.info("No existing model found. Training new model...")
        
        # Load dataset
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        try:
            data = pd.read_csv('heart.dat', sep='\s+', names=columns)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        # Prepare features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
        
        # Initialize SVM model
        base_model = SVC(probability=True, random_state=42)
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate on test set
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        # Save model and scaler
        dump(model, 'svm_model.joblib')
        dump(scaler, 'scaler.joblib')
        logger.info("Model and scaler saved successfully.")
        
        # Save evaluation metrics
        with open('model_metrics.txt', 'w') as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Best Parameters: {grid_search.best_params_}\n")
            f.write(f"Classification Report:\n{report}\n")
        
        return model, scaler

# Load model and scaler
try:
    model, scaler = train_model()
except Exception as e:
    logger.error(f"Error during model training: {e}")
    model, scaler = None, None

# Normalization parameters for reference
normalization_params = {
    'age': {'mean': 54.43, 'std': 9.04},
    'sex': {'mean': 0.68, 'std': 0.47},
    'cp': {'mean': 3.17, 'std': 0.95},
    'trestbps': {'mean': 131.69, 'std': 17.56},
    'chol': {'mean': 246.69, 'std': 51.78},
    'fbs': {'mean': 0.15, 'std': 0.36},
    'restecg': {'mean': 0.53, 'std': 0.53},
    'thalach': {'mean': 149.61, 'std': 22.88},
    'exang': {'mean': 0.33, 'std': 0.47},
    'oldpeak': {'mean': 1.04, 'std': 1.16},
    'slope': {'mean': 1.60, 'std': 0.62},
    'ca': {'mean': 0.68, 'std': 0.94},
    'thal': {'mean': 4.73, 'std': 1.94}
}

def analyze_risk_factors(form_data):
    """Analyze input data to identify risk factors based on clinical thresholds."""
    risk_factors = []
    
    try:
        if float(form_data['age']) > 60:
            risk_factors.append({
                'factor': 'Âge élevé',
                'level': 'high',
                'description': 'Âge > 60 ans'
            })
        
        if float(form_data['sex']) == 1:
            risk_factors.append({
                'factor': 'Sexe masculin',
                'level': 'medium',
                'description': 'Risque plus élevé chez les hommes'
            })
        
        if float(form_data['cp']) == 4:
            risk_factors.append({
                'factor': 'Asymptomatique',
                'level': 'high',
                'description': 'Absence de symptômes peut masquer la maladie'
            })
        
        if float(form_data['trestbps']) > 140:
            risk_factors.append({
                'factor': 'Hypertension',
                'level': 'high',
                'description': 'Pression artérielle élevée'
            })
        
        if float(form_data['chol']) > 240:
            risk_factors.append({
                'factor': 'Cholestérol élevé',
                'level': 'medium',
                'description': 'Cholestérol > 240 mg/dl'
            })
        
        if float(form_data['exang']) == 1:
            risk_factors.append({
                'factor': 'Angine d\'effort',
                'level': 'high',
                'description': 'Douleur thoracique à l\'exercice'
            })
        
        if float(form_data['ca']) > 0:
            risk_factors.append({
                'factor': 'Vaisseaux sténosés',
                'level': 'high',
                'description': f"{form_data['ca']} vaisseau(x) coloré(s)"
            })
        
        if float(form_data['thal']) == 7:
            risk_factors.append({
                'factor': 'Défaut de perfusion',
                'level': 'high',
                'description': 'Défaut réversible détecté'
            })
        
        return risk_factors
    except Exception as e:
        logger.error(f"Error in risk factor analysis: {e}")
        return []

def generate_recommendations(prediction, risk_factors, form_data):
    """Generate personalized recommendations based on prediction and risk factors."""
    recommendations = []
    
    try:
        if prediction == 2:  # High risk
            recommendations.extend([
                'Consultez immédiatement un cardiologue pour un bilan complet',
                'Effectuez des examens complémentaires (ECG, échocardiographie, test d\'effort)',
                'Surveillez régulièrement votre pression artérielle et votre rythme cardiaque'
            ])
        
        if float(form_data['chol']) > 240:
            recommendations.append('Adoptez un régime pauvre en cholestérol et riche en fibres')
        
        if float(form_data['trestbps']) > 140:
            recommendations.append('Réduisez votre consommation de sel et pratiquez une activité physique régulière')
        
        if float(form_data['age']) > 60:
            recommendations.append('Augmentez la fréquence de vos contrôles médicaux (tous les 6 mois)')
        
        recommendations.extend([
            'Maintenez un poids santé et évitez le tabac',
            'Pratiquez 30 minutes d\'exercice modéré 5 fois par semaine',
            'Gérez votre stress avec des techniques de relaxation'
        ])
        
        return recommendations
    except Exception as e:
        logger.error(f"Error in generating recommendations: {e}")
        return ['Consultez un professionnel de santé pour des recommandations personnalisées']

@app.route('/projet', methods=['GET'])
def projet():
    """Render the project HTML page."""
    return render_template('projet.html')

@app.route('/', methods=['GET'])
def home():
    """API home endpoint."""
    return jsonify({'message': 'Bienvenue sur l\'API de prédiction des maladies cardiaques ! Utilisez POST /predict pour faire des prédictions.'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict heart disease risk based on input features."""
    try:
        # Get JSON data from the frontend
        data = request.get_json()
        if not data:
            logger.error("No JSON data provided in request")
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing field: {field}")
                return jsonify({'error': f'Missing field: {field}'}), 400
            try:
                float(data[field])
            except ValueError:
                logger.error(f"Invalid value for {field}: {data[field]}")
                return jsonify({'error': f'Invalid value for {field}'}), 400
        
        # Convert data to array for prediction
        input_data = np.array([[
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
        ]])
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  # Probability of class 2 (disease)
        
        # Analyze risk factors and generate recommendations
        risk_factors = analyze_risk_factors(data)
        recommendations = generate_recommendations(prediction, risk_factors, data)
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }
        
        logger.info(f"Prediction made: {response}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)