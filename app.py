import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading Netflix dataset...")
df = pd.read_csv('netflix_cleaned_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset info:\n{df.info()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Data Preprocessing
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Create a copy for preprocessing
data = df.copy()

# Remove rows with missing RATING (target variable)
data = data[data['RATING'].notna() & (data['RATING'] != 'Not Rated')].copy()

# Convert RATING to numeric
data['RATING'] = pd.to_numeric(data['RATING'], errors='coerce')
data = data.dropna(subset=['RATING'])

print(f"Dataset after removing missing ratings: {data.shape}")
print(f"Rating range: {data['RATING'].min()} - {data['RATING'].max()}")

# Feature Engineering
# Extract duration in minutes for movies, extract season count for TV shows
data['DURATION_NUMERIC'] = data.apply(lambda row: 
    int(row['DURATION'].split()[0]) if pd.notna(row['DURATION']) and 'min' in str(row['DURATION']) 
    else (int(row['DURATION'].split()[0]) if pd.notna(row['DURATION']) and 'Season' in str(row['DURATION']) else 0),
    axis=1
)

# Encode TYPE (TV Show vs Movie)
data['TYPE_ENCODED'] = (data['TYPE'] == 'Movie').astype(int)

# Count number of cast members
data['CAST_COUNT'] = data['CAST'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

# Genre encoding (one-hot or label encoding)
le_genre = LabelEncoder()
data['GENRE_ENCODED'] = le_genre.fit_transform(data['GENRE'].fillna('Unknown'))

# Country encoding
le_country = LabelEncoder()
data['COUNTRY_ENCODED'] = le_country.fit_transform(data['COUNTRY'].fillna('Unknown'))

# Director availability
data['HAS_DIRECTOR'] = data['DIRECTOR'].notna().astype(int)

# Description availability
data['HAS_DESCRIPTION'] = data['DESCRIPTION'].notna().astype(int)

# Select features for modeling
feature_columns = [
    'TYPE_ENCODED', 
    'GENRE_ENCODED', 
    'DURATION_NUMERIC', 
    'COUNTRY_ENCODED', 
    'CAST_COUNT',
    'HAS_DIRECTOR',
    'HAS_DESCRIPTION'
]

X = data[feature_columns]
y = data['RATING']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Features used: {feature_columns}")

# Train-Test Split
print("\n" + "="*50)
print("TRAIN-TEST SPLIT")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Building and Training
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

# Train Random Forest Regressor
print("Training Random Forest Regressor...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("Model training completed!")

# Model Evaluation
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"Training R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"\nTraining RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"\nTraining MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Feature Importance
print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)

feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))

# Export Model and Artifacts
print("\n" + "="*50)
print("EXPORTING MODEL AND ARTIFACTS")
print("="*50)

# Save the model
model_path = 'netflix_rating_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

# Save the scaler
scaler_path = 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")

# Save label encoders
encoders = {
    'genre_encoder': le_genre,
    'country_encoder': le_country
}
encoders_path = 'encoders.pkl'
joblib.dump(encoders, encoders_path)
print(f"Encoders saved to: {encoders_path}")

# Save feature columns
feature_info = {
    'feature_columns': feature_columns,
    'feature_importance': feature_importance.to_dict('records')
}
info_path = 'feature_info.pkl'
joblib.dump(feature_info, info_path)
print(f"Feature info saved to: {info_path}")

# Create a summary report
print("\n" + "="*50)
print("MODEL SUMMARY REPORT")
print("="*50)

summary = f"""
NETFLIX RATING PREDICTION MODEL
{'='*50}

Dataset Information:
- Total records used: {len(data)}
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}

Model Configuration:
- Algorithm: Random Forest Regressor
- Number of trees: 100
- Max depth: 15
- Random state: 42

Performance Metrics:
- Training R² Score: {train_r2:.4f}
- Test R² Score: {test_r2:.4f}
- Training RMSE: {train_rmse:.4f}
- Test RMSE: {test_rmse:.4f}
- Training MAE: {train_mae:.4f}
- Test MAE: {test_mae:.4f}

Top 3 Important Features:
"""

for idx, row in feature_importance.head(3).iterrows():
    summary += f"  {row['Feature']}: {row['Importance']:.4f}\n"

summary += f"""
Exported Files:
- netflix_rating_model.pkl (trained model)
- scaler.pkl (feature scaler)
- encoders.pkl (label encoders)
- feature_info.pkl (feature columns and importance)

"""

print(summary)

# Save summary to file
with open('model_summary.txt', 'w') as f:
    f.write(summary)
print("Model summary saved to: model_summary.txt")

print("\nML Pipeline completed successfully!")
print("\nYou can now use the model for predictions using the saved files.")
