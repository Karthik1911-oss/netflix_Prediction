"""
Test script for Netflix Rating Prediction API
This script demonstrates how to use the Flask API to make predictions
"""

import requests
import json

# API Base URL
BASE_URL = 'http://localhost:5000'

def print_section(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)

# Test 1: Health Check
def test_health():
    print_section("TEST 1: Health Check")
    response = requests.get(f'{BASE_URL}/health')
    print(json.dumps(response.json(), indent=2))

# Test 2: Get Features Information
def test_features():
    print_section("TEST 2: Get Features Information")
    response = requests.get(f'{BASE_URL}/features')
    data = response.json()
    print(json.dumps(data, indent=2))

# Test 3: Single Prediction - Movie
def test_predict_movie():
    print_section("TEST 3: Predict Rating for a Movie")
    
    payload = {
        'type': 1,              # 1 = Movie, 0 = TV Show
        'genre': 'Drama',
        'duration': 120,        # 120 minutes for movie
        'country': 'USA',
        'cast_count': 5,
        'has_director': 1,
        'has_description': 1
    }
    
    print(f"Input: {json.dumps(payload, indent=2)}")
    response = requests.post(f'{BASE_URL}/predict', json=payload)
    print(f"\nOutput: {json.dumps(response.json(), indent=2)}")

# Test 4: Single Prediction - TV Show
def test_predict_tv_show():
    print_section("TEST 4: Predict Rating for a TV Show")
    
    payload = {
        'type': 0,              # 0 = TV Show
        'genre': 'Sci-Fi',
        'duration': 4,          # 4 seasons
        'country': 'Japan',
        'cast_count': 8,
        'has_director': 1,
        'has_description': 1
    }
    
    print(f"Input: {json.dumps(payload, indent=2)}")
    response = requests.post(f'{BASE_URL}/predict', json=payload)
    print(f"\nOutput: {json.dumps(response.json(), indent=2)}")

# Test 5: Batch Predictions
def test_batch_predictions():
    print_section("TEST 5: Batch Predictions")
    
    payload = {
        'predictions': [
            {
                'type': 1,
                'genre': 'Action',
                'duration': 150,
                'country': 'USA',
                'cast_count': 6,
                'has_director': 1,
                'has_description': 1
            },
            {
                'type': 0,
                'genre': 'Comedy',
                'duration': 3,
                'country': 'India',
                'cast_count': 10,
                'has_director': 1,
                'has_description': 0
            },
            {
                'type': 1,
                'genre': 'Romance',
                'duration': 90,
                'country': 'South Korea',
                'cast_count': 4,
                'has_director': 0,
                'has_description': 1
            }
        ]
    }
    
    print("Making batch predictions...")
    response = requests.post(f'{BASE_URL}/predict-batch', json=payload)
    print(f"Output: {json.dumps(response.json(), indent=2)}")

# Test 6: Invalid Input
def test_invalid_input():
    print_section("TEST 6: Invalid Input Error Handling")
    
    payload = {
        'type': 1,
        'genre': 'InvalidGenre',  # This genre doesn't exist
        'duration': 120,
        'country': 'USA',
        'cast_count': 5,
        'has_director': 1,
        'has_description': 1
    }
    
    print(f"Input with invalid genre: {json.dumps(payload, indent=2)}")
    response = requests.post(f'{BASE_URL}/predict', json=payload)
    print(f"\nOutput: {json.dumps(response.json(), indent=2)}")

# Test 7: Missing Fields
def test_missing_fields():
    print_section("TEST 7: Missing Required Fields")
    
    payload = {
        'type': 1,
        'genre': 'Drama'
        # Missing other required fields
    }
    
    print(f"Input with missing fields: {json.dumps(payload, indent=2)}")
    response = requests.post(f'{BASE_URL}/predict', json=payload)
    print(f"\nOutput: {json.dumps(response.json(), indent=2)}")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("NETFLIX RATING PREDICTION API - TEST SUITE")
    print("="*60)
    print(f"\nMake sure the Flask API is running at {BASE_URL}")
    print("Run: python api.py\n")
    
    try:
        # Run all tests
        test_health()
        test_features()
        test_predict_movie()
        test_predict_tv_show()
        test_batch_predictions()
        test_invalid_input()
        test_missing_fields()
        
        print_section("ALL TESTS COMPLETED")
        print("API is working correctly!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API")
        print("Make sure the Flask server is running: python api.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
