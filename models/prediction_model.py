"""
F1 Race Prediction Model
Implements machine learning algorithms for predicting F1 race results.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime
import json

from .data_processor import DataProcessor

logger = logging.getLogger(__name__)

class F1Predictor:
    """Main prediction model for F1 race results."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.data_processor = DataProcessor()
        self.models = {}
        self.feature_importance = {}
        self.is_trained = False
        self.model_path = model_path or 'models/saved_models/'
        
        # Initialize models
        self._initialize_models()
        
        # Load pre-trained models if available
        if os.path.exists(self.model_path):
            self._load_models()
    
    def _initialize_models(self):
        """Initialize different ML models for ensemble prediction."""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def train_models(self, data_path: Optional[str] = None, save_models: bool = True):
        """Train all prediction models on historical data."""
        try:
            # Load or create training data
            if data_path and os.path.exists(data_path):
                df = self.data_processor.load_historical_data(data_path)
            else:
                logger.info("Creating sample training data...")
                df = self.data_processor.create_sample_data()
            
            if df.empty:
                raise ValueError("No training data available")
            
            # Engineer features
            df_engineered = self.data_processor.engineer_features(df)
            
            # Prepare training data
            X, y = self._prepare_training_data(df_engineered)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train each model
            model_scores = {}
            for name, model in self.models.items():
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                model_scores[name] = {
                    'r2_score': score,
                    'mse': mse,
                    'mae': mae
                }
                
                # Store feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                
                logger.info(f"{name} - R²: {score:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
            
            self.is_trained = True
            
            # Save models if requested
            if save_models:
                self._save_models()
            
            # Save training results
            self._save_training_results(model_scores)
            
            logger.info("Model training completed successfully")
            return model_scores
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from engineered features."""
        try:
            # Extract features for each driver in each race
            X_list = []
            y_list = []
            
            for _, row in df.iterrows():
                # Get qualifying positions
                qualifying = row['qualifying_positions']
                
                # Get practice times
                fp1_times = row['practice_times_fp1']
                fp2_times = row['practice_times_fp2']
                fp3_times = row['practice_times_fp3']
                
                # Get race positions (target)
                race_positions = row['race_positions']
                
                # Create features for each driver
                for driver_idx in range(20):
                    # Prepare input for this driver
                    features = self.data_processor.prepare_prediction_input(
                        track=row['track'],
                        qualifying_positions=qualifying,
                        practice_results=[fp1_times, fp2_times, fp3_times],
                        weather_conditions=row['weather']
                    )
                    
                    # Add driver-specific features
                    driver_features = np.array([
                        qualifying[driver_idx],  # Driver's qualifying position
                        fp1_times[driver_idx] if fp1_times else 0,  # Driver's FP1 time
                        fp2_times[driver_idx] if fp2_times else 0,  # Driver's FP2 time
                        fp3_times[driver_idx] if fp3_times else 0,  # Driver's FP3 time
                    ])
                    
                    # Combine features
                    combined_features = np.concatenate([features.flatten(), driver_features])
                    X_list.append(combined_features)
                    
                    # Target: driver's race position
                    y_list.append(race_positions[driver_idx])
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def predict_race(self, 
                    track: str,
                    qualifying_positions: List[int],
                    practice_results: Optional[List[List[float]]] = None,
                    weather_conditions: str = 'dry') -> Dict:
        """Predict race results for given qualifying and practice data."""
        try:
            if not self.is_trained:
                logger.warning("Models not trained. Training with sample data...")
                self.train_models()
            
            # Validate input
            is_valid, message = self.data_processor.validate_input_data({
                'track': track,
                'qualifying_positions': qualifying_positions,
                'practice_results': practice_results,
                'weather_conditions': weather_conditions
            })
            
            if not is_valid:
                raise ValueError(message)
            
            # Prepare input features
            base_features = self.data_processor.prepare_prediction_input(
                track=track,
                qualifying_positions=qualifying_positions,
                practice_results=practice_results,
                weather_conditions=weather_conditions
            )
            
            # Make predictions for each driver
            predictions = []
            for driver_idx in range(20):
                # Add driver-specific features
                driver_features = np.array([
                    qualifying_positions[driver_idx],
                    practice_results[0][driver_idx] if practice_results and len(practice_results) > 0 else 0,
                    practice_results[1][driver_idx] if practice_results and len(practice_results) > 1 else 0,
                    practice_results[2][driver_idx] if practice_results and len(practice_results) > 2 else 0,
                ])
                
                # Combine features
                combined_features = np.concatenate([base_features.flatten(), driver_features])
                
                # Get predictions from all models
                model_predictions = {}
                for name, model in self.models.items():
                    try:
                        pred = model.predict(combined_features.reshape(1, -1))[0]
                        model_predictions[name] = max(1, round(pred))  # Ensure position >= 1
                    except Exception as e:
                        logger.warning(f"Error with {name} model: {e}")
                        model_predictions[name] = qualifying_positions[driver_idx]  # Fallback to qualifying
                
                # Ensemble prediction (weighted average)
                weights = {
                    'random_forest': 0.3,
                    'gradient_boosting': 0.3,
                    'linear_regression': 0.2,
                    'ridge_regression': 0.15,
                    'svr': 0.05
                }
                
                ensemble_pred = sum(
                    model_predictions[name] * weights[name]
                    for name in weights.keys()
                    if name in model_predictions
                )
                
                predictions.append({
                    'driver_position': driver_idx + 1,
                    'qualifying_position': qualifying_positions[driver_idx],
                    'predicted_race_position': max(1, round(ensemble_pred)),
                    'model_predictions': model_predictions,
                    'confidence': self._calculate_confidence(model_predictions)
                })
            
            # Sort by predicted race position
            predictions.sort(key=lambda x: x['predicted_race_position'])
            
            # Add metadata
            result = {
                'track': track,
                'weather_conditions': weather_conditions,
                'prediction_timestamp': datetime.now().isoformat(),
                'predictions': predictions,
                'model_ensemble': list(self.models.keys()),
                'overall_confidence': np.mean([p['confidence'] for p in predictions])
            }
            
            logger.info(f"Generated predictions for {track} with {len(predictions)} drivers")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def _calculate_confidence(self, model_predictions: Dict[str, float]) -> float:
        """Calculate confidence score based on model agreement."""
        if not model_predictions:
            return 0.0
        
        predictions = list(model_predictions.values())
        std_dev = np.std(predictions)
        mean_pred = np.mean(predictions)
        
        # Higher confidence when models agree (lower std dev)
        confidence = max(0.0, 1.0 - (std_dev / mean_pred))
        return min(1.0, confidence)
    
    def get_feature_importance(self) -> Dict[str, List[float]]:
        """Get feature importance from trained models."""
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        feature_names = self.data_processor.get_feature_names()
        
        for model_name, importance in self.feature_importance.items():
            if len(importance) == len(feature_names):
                importance_dict[model_name] = {
                    'features': feature_names,
                    'importance': importance.tolist()
                }
        
        return importance_dict
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            for name, model in self.models.items():
                model_file = os.path.join(self.model_path, f"{name}.joblib")
                joblib.dump(model, model_file)
                logger.info(f"Saved {name} model to {model_file}")
            
            # Save feature importance
            importance_file = os.path.join(self.model_path, "feature_importance.json")
            with open(importance_file, 'w') as f:
                json.dump(self.feature_importance, f, default=lambda x: x.tolist())
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load pre-trained models from disk."""
        try:
            for name in self.models.keys():
                model_file = os.path.join(self.model_path, f"{name}.joblib")
                if os.path.exists(model_file):
                    self.models[name] = joblib.load(model_file)
                    logger.info(f"Loaded {name} model from {model_file}")
            
            # Load feature importance
            importance_file = os.path.join(self.model_path, "feature_importance.json")
            if os.path.exists(importance_file):
                with open(importance_file, 'r') as f:
                    self.feature_importance = json.load(f)
            
            self.is_trained = True
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _save_training_results(self, results: Dict):
        """Save training results to file."""
        try:
            results_file = os.path.join(self.model_path, "training_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved training results to {results_file}")
        except Exception as e:
            logger.error(f"Error saving training results: {e}")
    
    def get_model_performance(self) -> Dict:
        """Get current model performance metrics."""
        if not self.is_trained:
            return {'status': 'Models not trained'}
        
        try:
            results_file = os.path.join(self.model_path, "training_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    return json.load(f)
            else:
                return {'status': 'No performance data available'}
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            return {'status': 'Error loading performance data'} 