import os
from flask import Flask, request, jsonify, Response
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load pre-trained model and artifacts
print("Loading model and artifacts...")
model = joblib.load('netflix_rating_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')
feature_info = joblib.load('feature_info.pkl')

genre_encoder = encoders['genre_encoder']
country_encoder = encoders['country_encoder']
feature_columns = feature_info['feature_columns']

print("Model and artifacts loaded successfully!")

# Home route
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Netflix Rating Prediction API',
        'version': '1.0',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'POST /predict': 'Predict Netflix rating',
            'GET /features': 'Get feature information'
        }
    })

# Health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'features': feature_columns
    })

# Feature information route
@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({
        'required_features': feature_columns,
        'feature_descriptions': {
            'type': 'TV Show (0) or Movie (1)',
            'genre': 'Genre name (e.g., Drama, Action, Comedy)',
            'duration': 'Duration in minutes (for movies) or number of seasons (for TV shows)',
            'country': 'Country name (e.g., USA, India, Japan)',
            'cast_count': 'Number of cast members',
            'has_director': 'Has director info (0 or 1)',
            'has_description': 'Has description (0 or 1)'
        },
        'supported_genres': list(genre_encoder.classes_),
        'supported_countries': list(country_encoder.classes_)
    })

# Prediction route - HTML Form for GET requests
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Handle GET requests - serve an HTML form
    if request.method == 'GET':
        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Netflix Rating Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; background: #f0f0f0; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #e50914; text-align: center; }
        .form-group { margin: 15px 0; }
        label { display: block; font-weight: bold; margin-bottom: 5px; color: #333; }
        input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; box-sizing: border-box; }
        button { background: #e50914; color: white; padding: 12px 30px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; width: 100%; margin-top: 20px; font-weight: bold; }
        button:hover { background: #b20710; }
        .result { margin-top: 20px; padding: 15px; background: #f9f9f9; border-radius: 4px; display: none; border-left: 4px solid #e50914; }
        .result.show { display: block; }
        .success { color: #28a745; }
        .error { color: #dc3545; }
        .info { color: #666; font-size: 14px; text-align: center; margin-bottom: 20px; }
        .rating-display { font-size: 32px; font-weight: bold; color: #e50914; text-align: center; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Netflix Rating Predictor</h1>
        <p class="info">Fill in the details to predict a Netflix content rating</p>
        
        <form id="predictForm" onsubmit="return predictRating(event)">
            <div class="form-group">
                <label for="type">Content Type:</label>
                <select id="type" required>
                    <option value="1">Movie</option>
                    <option value="0">TV Show</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="genre">Genre:</label>
                <select id="genre" required>
                    <option value="">Select Genre</option>
                    <option value="Action">Action</option>
                    <option value="Comedy">Comedy</option>
                    <option value="Crime">Crime</option>
                    <option value="Documentary">Documentary</option>
                    <option value="Drama">Drama</option>
                    <option value="Horror">Horror</option>
                    <option value="Romance">Romance</option>
                    <option value="Sci-Fi">Sci-Fi</option>
                    <option value="Thriller">Thriller</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="duration">Duration (minutes for movie / seasons for TV show):</label>
                <input type="number" id="duration" value="120" required min="1">
            </div>
            
            <div class="form-group">
                <label for="country">Country:</label>
                <select id="country" required>
                    <option value="">Select Country</option>
                    <option value="USA">USA</option>
                    <option value="India">India</option>
                    <option value="Japan">Japan</option>
                    <option value="South Korea">South Korea</option>
                    <option value="UK">UK</option>
                    <option value="Canada">Canada</option>
                    <option value="France">France</option>
                    <option value="Germany">Germany</option>
                    <option value="Spain">Spain</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="cast_count">Number of Cast Members:</label>
                <input type="number" id="cast_count" value="5" required min="0">
            </div>
            
            <div class="form-group">
                <label for="has_director">Has Director:</label>
                <select id="has_director" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="has_description">Has Description:</label>
                <select id="has_description" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            
            <button type="submit">🔮 Predict Rating</button>
        </form>
        
        <div id="result" class="result">
            <div id="resultContent"></div>
        </div>
    </div>
    
    <script>
        async function predictRating(event) {
            event.preventDefault();
            
            const formData = {
                type: parseInt(document.getElementById('type').value),
                genre: document.getElementById('genre').value,
                duration: parseInt(document.getElementById('duration').value),
                country: document.getElementById('country').value,
                cast_count: parseInt(document.getElementById('cast_count').value),
                has_director: parseInt(document.getElementById('has_director').value),
                has_description: parseInt(document.getElementById('has_description').value)
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                if (data.success) {
                    resultContent.innerHTML = `
                        <div class="success">
                            <h3>✓ Prediction Successful</h3>
                            <p><strong>Content Type:</strong> ${data.prediction.input_features.type}</p>
                            <p><strong>Genre:</strong> ${data.prediction.input_features.genre}</p>
                            <p><strong>Country:</strong> ${data.prediction.input_features.country}</p>
                            <p><strong>Cast Members:</strong> ${data.prediction.input_features.cast_count}</p>
                            <div class="rating-display">${data.prediction.predicted_rating}/10</div>
                        </div>
                    `;
                } else {
                    resultContent.innerHTML = `<div class="error"><h3>✗ Prediction Failed</h3><p>${data.error || 'Unknown error'}</p></div>`;
                }
                resultDiv.classList.add('show');
            } catch (error) {
                document.getElementById('resultContent').innerHTML = `<div class="error"><p>Error: ${error.message}</p></div>`;
                document.getElementById('result').classList.add('show');
            }
            
            return false;
        }
    </script>
</body>
</html>'''
        return Response(html_content, mimetype='text/html')
    
    try:
        # Get JSON data from request (POST)
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['type', 'genre', 'duration', 'country', 'cast_count', 'has_director', 'has_description']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields,
                'required_fields': required_fields
            }), 400
        
        # Extract and validate input data
        try:
            content_type = int(data['type'])  # 0 for TV Show, 1 for Movie
            genre = str(data['genre']).strip()
            duration = int(data['duration'])
            country = str(data['country']).strip()
            cast_count = int(data['cast_count'])
            has_director = int(data['has_director'])
            has_description = int(data['has_description'])
            
            # Validate ranges
            if content_type not in [0, 1]:
                return jsonify({'error': 'type must be 0 (TV Show) or 1 (Movie)'}), 400
            
            if duration < 0:
                return jsonify({'error': 'duration must be positive'}), 400
            
            if cast_count < 0:
                return jsonify({'error': 'cast_count must be non-negative'}), 400
            
            if has_director not in [0, 1]:
                return jsonify({'error': 'has_director must be 0 or 1'}), 400
            
            if has_description not in [0, 1]:
                return jsonify({'error': 'has_description must be 0 or 1'}), 400
            
        except ValueError as e:
            return jsonify({'error': f'Invalid input type: {str(e)}'}), 400
        
        # Encode categorical features
        try:
            genre_encoded = genre_encoder.transform([genre])[0]
        except ValueError:
            return jsonify({
                'error': f'Invalid genre: {genre}',
                'supported_genres': list(genre_encoder.classes_)
            }), 400
        
        try:
            country_encoded = country_encoder.transform([country])[0]
        except ValueError:
            return jsonify({
                'error': f'Invalid country: {country}',
                'supported_countries': list(country_encoder.classes_)
            }), 400
        
        # Create feature vector in the correct order
        features = np.array([[
            content_type,           # TYPE_ENCODED
            genre_encoded,          # GENRE_ENCODED
            duration,               # DURATION_NUMERIC
            country_encoded,        # COUNTRY_ENCODED
            cast_count,             # CAST_COUNT
            has_director,           # HAS_DIRECTOR
            has_description         # HAS_DESCRIPTION
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predicted_rating = model.predict(features_scaled)[0]
        
        # Ensure rating is within reasonable bounds (clamp between 0 and 10)
        predicted_rating = max(0, min(10, predicted_rating))
        
        return jsonify({
            'prediction': {
                'predicted_rating': round(predicted_rating, 2),
                'confidence': 'Medium',
                'input_features': {
                    'type': 'Movie' if content_type == 1 else 'TV Show',
                    'genre': genre,
                    'duration': duration,
                    'country': country,
                    'cast_count': cast_count,
                    'has_director': bool(has_director),
                    'has_description': bool(has_description)
                }
            },
            'success': True
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'success': False
        }), 500

# Batch prediction route
@app.route('/predict-batch', methods=['GET', 'POST'])
def predict_batch():
    # Handle GET requests - serve an HTML form for batch predictions
    if request.method == 'GET':
        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Netflix Batch Rating Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 50px auto; background: #f0f0f0; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #e50914; text-align: center; }
        .controls { margin: 20px 0; text-align: center; }
        button { background: #e50914; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        button:hover { background: #b20710; }
        .predictions-container { margin-top: 20px; }
        .prediction-item { background: #f9f9f9; padding: 15px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; }
        .prediction-item h4 { margin-top: 0; color: #333; }
        .form-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
        input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 12px; box-sizing: border-box; }
        label { display: block; font-size: 12px; font-weight: bold; margin-bottom: 3px; color: #333; }
        .result { margin-top: 20px; padding: 15px; background: #f9f9f9; border-radius: 4px; display: none; border-left: 4px solid #e50914; }
        .result.show { display: block; }
        .result-table { width: 100%; border-collapse: collapse; }
        .result-table th, .result-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        .result-table th { background: #e50914; color: white; }
        .success { color: green; }
        .error { color: red; }
        .info { color: #666; font-size: 14px; text-align: center; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Netflix Batch Rating Predictor</h1>
        <p class="info">Predict ratings for multiple Netflix contents at once</p>
        
        <div class="controls">
            <button onclick="addPrediction()">+ Add Content</button>
            <button onclick="clearAll()" style="background: #666;">Clear All</button>
            <button onclick="submitBatch()" style="background: #28a745;">🔮 Predict All Ratings</button>
        </div>
        
        <div id="predictionsContainer" class="predictions-container"></div>
        
        <div id="result" class="result">
            <div id="resultContent"></div>
        </div>
    </div>
    
    <script>
        let predictionCount = 0;
        
        function addPrediction() {
            const container = document.getElementById('predictionsContainer');
            predictionCount++;
            
            const html = `
                <div class="prediction-item" id="pred-${predictionCount}">
                    <h4>Content #${predictionCount} <button type="button" onclick="removePrediction(${predictionCount})" style="background: red; padding: 5px 10px; float: right;">Remove</button></h4>
                    <div class="form-row">
                        <div>
                            <label>Type:</label>
                            <select id="type-${predictionCount}">
                                <option value="1">Movie</option>
                                <option value="0">TV Show</option>
                            </select>
                        </div>
                        <div>
                            <label>Genre:</label>
                            <select id="genre-${predictionCount}">
                                <option value="">Select</option>
                                <option value="Action">Action</option>
                                <option value="Comedy">Comedy</option>
                                <option value="Drama">Drama</option>
                                <option value="Horror">Horror</option>
                                <option value="Romance">Romance</option>
                                <option value="Sci-Fi">Sci-Fi</option>
                                <option value="Thriller">Thriller</option>
                            </select>
                        </div>
                        <div>
                            <label>Duration (min/seasons):</label>
                            <input type="number" id="duration-${predictionCount}" value="120" min="1">
                        </div>
                    </div>
                    <div class="form-row">
                        <div>
                            <label>Country:</label>
                            <select id="country-${predictionCount}">
                                <option value="">Select</option>
                                <option value="USA">USA</option>
                                <option value="India">India</option>
                                <option value="Japan">Japan</option>
                                <option value="South Korea">South Korea</option>
                                <option value="UK">UK</option>
                            </select>
                        </div>
                        <div>
                            <label>Cast Count:</label>
                            <input type="number" id="cast_count-${predictionCount}" value="5" min="0">
                        </div>
                        <div>
                            <label>Has Director:</label>
                            <select id="has_director-${predictionCount}">
                                <option value="1">Yes</option>
                                <option value="0">No</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div>
                            <label>Has Description:</label>
                            <select id="has_description-${predictionCount}">
                                <option value="1">Yes</option>
                                <option value="0">No</option>
                            </select>
                        </div>
                    </div>
                </div>
            `;
            container.innerHTML += html;
        }
        
        function removePrediction(id) {
            const elem = document.getElementById(`pred-${id}`);
            if (elem) elem.remove();
        }
        
        function clearAll() {
            document.getElementById('predictionsContainer').innerHTML = '';
            predictionCount = 0;
        }
        
        async function submitBatch() {
            const predictions = [];
            const container = document.getElementById('predictionsContainer');
            const items = container.querySelectorAll('.prediction-item');
            
            if (items.length === 0) {
                alert('Please add at least one content');
                return;
            }
            
            for (const item of items) {
                const id = item.id.replace('pred-', '');
                predictions.push({
                    type: parseInt(document.getElementById(`type-${id}`).value),
                    genre: document.getElementById(`genre-${id}`).value,
                    duration: parseInt(document.getElementById(`duration-${id}`).value),
                    country: document.getElementById(`country-${id}`).value,
                    cast_count: parseInt(document.getElementById(`cast_count-${id}`).value),
                    has_director: parseInt(document.getElementById(`has_director-${id}`).value),
                    has_description: parseInt(document.getElementById(`has_description-${id}`).value)
                });
            }
            
            try {
                const response = await fetch('/predict-batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ predictions })
                });
                
                const data = await response.json();
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                if (data.batch_results) {
                    let tableHtml = `<h3>Batch Prediction Results (${data.successful}/${data.total_items} successful)</h3>
                        <table class="result-table">
                            <thead>
                                <tr>
                                    <th>Content #</th>
                                    <th>Predicted Rating</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>`;
                    
                    for (const result of data.batch_results) {
                        const status = result.success ? `<span class="success">✓ Success</span>` : `<span class="error">✗ Error: ${result.error}</span>`;
                        const rating = result.success ? `${result.predicted_rating}/10` : 'N/A';
                        tableHtml += `<tr><td>#${result.item_index + 1}</td><td>${rating}</td><td>${status}</td></tr>`;
                    }
                    
                    tableHtml += '</tbody></table>';
                    resultContent.innerHTML = tableHtml;
                } else {
                    resultContent.innerHTML = `<div class="error"><h3>Error</h3><p>${data.error || 'Unknown error'}</p></div>`;
                }
                resultDiv.classList.add('show');
            } catch (error) {
                document.getElementById('resultContent').innerHTML = `<div class="error"><p>Error: ${error.message}</p></div>`;
                document.getElementById('result').classList.add('show');
            }
        }
        
        // Add first prediction by default
        addPrediction();
    </script>
</body>
</html>'''
        return Response(html_content, mimetype='text/html')
    
    try:
        data = request.get_json()
        
        if 'predictions' not in data or not isinstance(data['predictions'], list):
            return jsonify({'error': 'Expected list of predictions in request body'}), 400
        
        results = []
        
        for idx, item in enumerate(data['predictions']):
            required_fields = ['type', 'genre', 'duration', 'country', 'cast_count', 'has_director', 'has_description']
            
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                results.append({
                    'item_index': idx,
                    'error': f'Missing fields: {missing_fields}',
                    'success': False
                })
                continue
            
            try:
                content_type = int(item['type'])
                genre = str(item['genre']).strip()
                duration = int(item['duration'])
                country = str(item['country']).strip()
                cast_count = int(item['cast_count'])
                has_director = int(item['has_director'])
                has_description = int(item['has_description'])
                
                genre_encoded = genre_encoder.transform([genre])[0]
                country_encoded = country_encoder.transform([country])[0]
                
                features = np.array([[
                    content_type,
                    genre_encoded,
                    duration,
                    country_encoded,
                    cast_count,
                    has_director,
                    has_description
                ]])
                
                features_scaled = scaler.transform(features)
                predicted_rating = model.predict(features_scaled)[0]
                predicted_rating = max(0, min(10, predicted_rating))
                
                results.append({
                    'item_index': idx,
                    'predicted_rating': round(predicted_rating, 2),
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'item_index': idx,
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'batch_results': results,
            'total_items': len(data['predictions']),
            'successful': sum(1 for r in results if r['success'])
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction error: {str(e)}',
            'success': False
        }), 500

# Error handler for 404
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /',
            'GET /health',
            'GET /features',
            'POST /predict',
            'POST /predict-batch'
        ]
    }), 404

# Error handler for 405
@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed'
    }), 405

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("Starting Flask API server...")
    print("Available endpoints:")
    print("  GET  / - API information")
    print("  GET  /health - Health check")
    print("  GET  /features - Feature information")
    print("  POST /predict - Make single prediction")
    print("  POST /predict-batch - Make batch predictions")
    print(f"\nServer running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
