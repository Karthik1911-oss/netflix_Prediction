# Netflix Rating Prediction API

A Flask REST API for predicting Netflix content ratings using a trained Machine Learning model (Random Forest).

## 📋 Prerequisites

- Python 3.7+
- All packages listed in `requirements.txt`

## 🚀 Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure model files are present:**
The following files must be in the same directory:
- `netflix_rating_model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `encoders.pkl` - Categorical encoders
- `feature_info.pkl` - Feature information

## ▶️ Running the API

```bash
python api.py
```

The API will start on `http://localhost:5000`

## 📡 API Endpoints

### 1. **GET** `/`
Get API information and available endpoints.

**Response:**
```json
{
  "message": "Netflix Rating Prediction API",
  "version": "1.0",
  "endpoints": { ... }
}
```

### 2. **GET** `/health`
Health check endpoint to verify the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "features": [...]
}
```

### 3. **GET** `/features`
Get information about all supported features, genres, and countries.

**Response:**
```json
{
  "required_features": ["type", "genre", "duration", "country", "cast_count", "has_director", "has_description"],
  "feature_descriptions": { ... },
  "supported_genres": [...],
  "supported_countries": [...]
}
```

### 4. **POST** `/predict`
Make a single prediction for a Netflix content.

**Request Body:**
```json
{
  "type": 1,
  "genre": "Drama",
  "duration": 120,
  "country": "USA",
  "cast_count": 5,
  "has_director": 1,
  "has_description": 1
}
```

**Field Descriptions:**
- `type` (int): 0 = TV Show, 1 = Movie
- `genre` (str): Genre name (see `/features` endpoint)
- `duration` (int): Minutes for movies, Seasons for TV shows
- `country` (str): Country name (see `/features` endpoint)
- `cast_count` (int): Number of cast members
- `has_director` (int): 0 or 1 (has director or not)
- `has_description` (int): 0 or 1 (has description or not)

**Response:**
```json
{
  "prediction": {
    "predicted_rating": 7.45,
    "confidence": "Medium",
    "input_features": { ... }
  },
  "success": true
}
```

### 5. **POST** `/predict-batch`
Make multiple predictions at once.

**Request Body:**
```json
{
  "predictions": [
    {
      "type": 1,
      "genre": "Drama",
      "duration": 120,
      "country": "USA",
      "cast_count": 5,
      "has_director": 1,
      "has_description": 1
    },
    {
      "type": 0,
      "genre": "Comedy",
      "duration": 3,
      "country": "India",
      "cast_count": 10,
      "has_director": 1,
      "has_description": 0
    }
  ]
}
```

**Response:**
```json
{
  "batch_results": [
    {
      "item_index": 0,
      "predicted_rating": 7.45,
      "success": true
    },
    {
      "item_index": 1,
      "predicted_rating": 6.82,
      "success": true
    }
  ],
  "total_items": 2,
  "successful": 2
}
```

## 🧪 Testing the API

### Using Python (Recommended)
```bash
python test_api.py
```

This runs comprehensive tests including:
- Health checks
- Feature information retrieval
- Single predictions (Movie and TV Show)
- Batch predictions
- Error handling

### Using cURL

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": 1,
    "genre": "Drama",
    "duration": 120,
    "country": "USA",
    "cast_count": 5,
    "has_director": 1,
    "has_description": 1
  }'
```

**Get Features:**
```bash
curl http://localhost:5000/features
```

**Get Health Status:**
```bash
curl http://localhost:5000/health
```

### Using Postman

1. Open Postman
2. Create a new POST request
3. URL: `http://localhost:5000/predict`
4. Headers: `Content-Type: application/json`
5. Body (raw JSON):
```json
{
  "type": 1,
  "genre": "Drama",
  "duration": 120,
  "country": "USA",
  "cast_count": 5,
  "has_director": 1,
  "has_description": 1
}
```

## 📊 Example Predictions

### Example 1: Hollywood Action Movie
```json
{
  "type": 1,
  "genre": "Action",
  "duration": 150,
  "country": "USA",
  "cast_count": 10,
  "has_director": 1,
  "has_description": 1
}
```
**Expected Rating:** ~7.2-7.8

### Example 2: Indian Web Series
```json
{
  "type": 0,
  "genre": "Drama",
  "duration": 5,
  "country": "India",
  "cast_count": 15,
  "has_director": 1,
  "has_description": 1
}
```
**Expected Rating:** ~6.5-7.0

### Example 3: Japanese Anime
```json
{
  "type": 0,
  "genre": "Sci-Fi",
  "duration": 3,
  "country": "Japan",
  "cast_count": 5,
  "has_director": 0,
  "has_description": 1
}
```
**Expected Rating:** ~6.8-7.3

## ⚠️ Error Handling

The API provides detailed error messages for common issues:

1. **Missing Fields:**
```json
{
  "error": "Missing required fields",
  "missing_fields": ["duration", "cast_count"],
  "required_fields": [...]
}
```

2. **Invalid Genre:**
```json
{
  "error": "Invalid genre: InvalidGenre",
  "supported_genres": [...]
}
```

3. **Invalid Country:**
```json
{
  "error": "Invalid country: InvalidCountry",
  "supported_countries": [...]
}
```

4. **Invalid Data Type:**
```json
{
  "error": "Invalid input type: could not convert string to float"
}
```

## 📁 Project Structure

```
├── app.py                          # ML model training script
├── api.py                          # Flask API server
├── test_api.py                     # API test suite
├── requirements.txt                # Python dependencies
├── netflix_cleaned_dataset.csv     # Training dataset
├── netflix_rating_model.pkl        # Trained model (exported)
├── scaler.pkl                      # Feature scaler (exported)
├── encoders.pkl                    # Categorical encoders (exported)
├── feature_info.pkl                # Feature information (exported)
├── model_summary.txt               # Model performance report
└── README.md                       # This file
```

## 🎯 Model Information

- **Algorithm:** Random Forest Regressor
- **Number of Trees:** 100
- **Features Used:** 7 (Type, Genre, Duration, Country, Cast Count, Director, Description)
- **Training Samples:** 1,272
- **Test Samples:** 318
- **Top Feature:** Genre (29.24% importance)

## 🔧 Troubleshooting

### Port 5000 Already in Use
```bash
# Change port in api.py (last line):
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Model Files Not Found
- Ensure all `.pkl` files are in the same directory as `api.py`
- Run `python app.py` to regenerate model files if needed

### Dependency Issues
```bash
pip install --upgrade -r requirements.txt
```

## 📝 API Documentation

For more details on features and predictions:
- Visit `http://localhost:5000/features` for supported values
- Check `model_summary.txt` for model performance metrics
- See `test_api.py` for example usage patterns

## 🚀 Deployment

### Production Setup (using Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

### Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "api.py"]
```

Build and run:
```bash
docker build -t netflix-api .
docker run -p 5000:5000 netflix-api
```

## 📄 License

MIT License

## 👤 Author

Your Name / Your Organization

---

**Last Updated:** 2026-06-17
