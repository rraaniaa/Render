from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from joblib import load, dump

app = Flask(__name__)
CORS(app)

# Load or train the SVM model
try:
    model = load('svm_model.joblib')
    scaler = load('scaler.joblib')
except FileNotFoundError:
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    data = pd.read_csv('heart.dat', sep='\s+', names=columns)
    X = data.drop('target', axis=1)
    y = data['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(kernel='linear', C=0.1, gamma='scale', probability=True, random_state=42)
    model.fit(X_scaled, y)
    dump(model, 'svm_model.joblib')
    dump(scaler, 'scaler.joblib')

# Normalization parameters from your EDA (for reference in risk factor analysis)
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
    risk_factors = []
    
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

def generate_recommendations(prediction, risk_factors, form_data):
    recommendations = []
    
    if prediction == 2:  # High risk
        recommendations.extend([
            'Consultez immédiatement un cardiologue pour un bilan complet',
            'Effectuez des examens complémentaires (ECG, échocardiographie, test d\'effort)',
            'Surveillez régulièrement votre pression artérielle et votre rythme cardiaque'
        ])
    
    if float(form_data['chol']) > 240:
        recommendations.append('Adoptez un régime pauvre en cholestérol et riches en fibres')
    
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

@app.route('/projet', methods=['GET'])
def projet():
    return render_template('projet.html')


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Bienvenue sur l\'API de prédiction des maladies cardiaques ! Utilisez POST /predict pour faire des prédictions.'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the frontend
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
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
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)