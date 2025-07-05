"""
Data Processor for F1 Race Predictions
Handles data preprocessing, feature engineering, and data validation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging
from typing import List, Dict, Tuple, Optional
import json
import os

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles all data processing operations for F1 race predictions."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_names = []
        self.is_fitted = False
        
        # Track-specific features
        self.track_features = {
            'Monaco': {'circuit_type': 'street', 'avg_speed': 160, 'corners': 19},
            'Silverstone': {'circuit_type': 'permanent', 'avg_speed': 180, 'corners': 18},
            'Monza': {'circuit_type': 'permanent', 'avg_speed': 220, 'corners': 11},
            'Spa-Francorchamps': {'circuit_type': 'permanent', 'avg_speed': 190, 'corners': 20},
            'Suzuka': {'circuit_type': 'permanent', 'avg_speed': 170, 'corners': 18},
            'Interlagos': {'circuit_type': 'permanent', 'avg_speed': 175, 'corners': 15},
            'Red Bull Ring': {'circuit_type': 'permanent', 'avg_speed': 185, 'corners': 10},
            'Hungaroring': {'circuit_type': 'permanent', 'avg_speed': 165, 'corners': 14},
            'Circuit de Catalunya': {'circuit_type': 'permanent', 'avg_speed': 175, 'corners': 16},
            'Albert Park': {'circuit_type': 'street', 'avg_speed': 170, 'corners': 16},
            'Jeddah Corniche Circuit': {'circuit_type': 'street', 'avg_speed': 185, 'corners': 27},
            'Baku City Circuit': {'circuit_type': 'street', 'avg_speed': 175, 'corners': 20}
        }
        
        # Weather impact factors
        self.weather_factors = {
            'dry': 1.0,
            'wet': 0.8,
            'intermediate': 0.9,
            'mixed': 0.85
        }
    
    def load_historical_data(self, file_path: str) -> pd.DataFrame:
        """Load historical F1 data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample historical data for demonstration purposes."""
        np.random.seed(42)
        
        # Generate sample data
        n_races = 500
        tracks = list(self.track_features.keys())
        drivers = [
            "Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "Lando Norris",
            "Carlos Sainz", "George Russell", "Fernando Alonso", "Oscar Piastri"
        ]
        
        data = []
        for i in range(n_races):
            track = np.random.choice(tracks)
            weather = np.random.choice(['dry', 'wet', 'intermediate'], p=[0.7, 0.2, 0.1])
            
            # Generate qualifying positions (1-20)
            qualifying_positions = list(range(1, 21))
            np.random.shuffle(qualifying_positions)
            
            # Generate practice results (lap times in seconds)
            base_lap_time = self.track_features[track]['avg_speed'] / 3.6  # Convert to seconds
            practice_times = []
            for j in range(3):  # 3 practice sessions
                session_times = []
                for driver in range(20):
                    # Add some randomness to lap times
                    time = base_lap_time + np.random.normal(0, 2) + np.random.normal(0, 1) * (qualifying_positions[driver] - 1)
                    session_times.append(max(time, base_lap_time * 0.8))  # Ensure reasonable times
                practice_times.append(session_times)
            
            # Generate race results (with some correlation to qualifying)
            race_positions = qualifying_positions.copy()
            # Add some randomness to race results
            for j in range(len(race_positions)):
                if np.random.random() < 0.3:  # 30% chance of position change
                    swap_idx = np.random.randint(0, len(race_positions))
                    race_positions[j], race_positions[swap_idx] = race_positions[swap_idx], race_positions[j]
            
            data.append({
                'race_id': i + 1,
                'track': track,
                'weather': weather,
                'qualifying_positions': qualifying_positions,
                'practice_times_fp1': practice_times[0],
                'practice_times_fp2': practice_times[1],
                'practice_times_fp3': practice_times[2],
                'race_positions': race_positions,
                'season': np.random.randint(2018, 2024)
            })
        
        return pd.DataFrame(data)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw F1 data."""
        try:
            df_engineered = df.copy()
            
            # Track features
            df_engineered['circuit_type'] = df_engineered['track'].map(
                {track: features['circuit_type'] for track, features in self.track_features.items()}
            )
            df_engineered['avg_speed'] = df_engineered['track'].map(
                {track: features['avg_speed'] for track, features in self.track_features.items()}
            )
            df_engineered['corners'] = df_engineered['track'].map(
                {track: features['corners'] for track, features in self.track_features.items()}
            )
            
            # Weather features
            df_engineered['weather_factor'] = df_engineered['weather'].map(self.weather_factors)
            
            # Qualifying features
            df_engineered['qualifying_gap_to_pole'] = df_engineered['qualifying_positions'].apply(
                lambda x: [pos - 1 for pos in x]  # Gap to pole position
            )
            df_engineered['qualifying_consistency'] = df_engineered['qualifying_positions'].apply(
                lambda x: np.std(x[:10])  # Consistency of top 10
            )
            
            # Practice features
            df_engineered['practice_avg_time'] = df_engineered.apply(
                lambda row: np.mean([
                    np.mean(row['practice_times_fp1']),
                    np.mean(row['practice_times_fp2']),
                    np.mean(row['practice_times_fp3'])
                ]), axis=1
            )
            df_engineered['practice_consistency'] = df_engineered.apply(
                lambda row: np.std([
                    np.std(row['practice_times_fp1']),
                    np.std(row['practice_times_fp2']),
                    np.std(row['practice_times_fp3'])
                ]), axis=1
            )
            
            # Race outcome features
            df_engineered['position_change'] = df_engineered.apply(
                lambda row: [race - qual for race, qual in zip(row['race_positions'], row['qualifying_positions'])], axis=1
            )
            df_engineered['top_5_accuracy'] = df_engineered.apply(
                lambda row: sum(1 for i in range(5) if row['race_positions'][i] <= 5) / 5, axis=1
            )
            
            # Season features
            df_engineered['season_progress'] = (df_engineered['season'] - 2010) / 14  # Normalized season progress
            
            logger.info(f"Engineered {len(df_engineered.columns)} features from {len(df_engineered)} records")
            return df_engineered
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return df
    
    def prepare_prediction_input(self, 
                                track: str,
                                qualifying_positions: List[int],
                                practice_results: Optional[List[List[float]]] = None,
                                weather_conditions: str = 'dry') -> np.ndarray:
        """Prepare input data for prediction."""
        try:
            # Validate inputs
            if len(qualifying_positions) != 20:
                raise ValueError("Must provide qualifying positions for all 20 drivers")
            
            if track not in self.track_features:
                raise ValueError(f"Unknown track: {track}")
            
            if weather_conditions not in self.weather_factors:
                raise ValueError(f"Unknown weather condition: {weather_conditions}")
            
            # Create feature vector
            features = []
            
            # Track features
            track_info = self.track_features[track]
            features.extend([
                1 if track_info['circuit_type'] == 'street' else 0,  # Street circuit
                1 if track_info['circuit_type'] == 'permanent' else 0,  # Permanent circuit
                track_info['avg_speed'] / 250,  # Normalized average speed
                track_info['corners'] / 30,  # Normalized corner count
            ])
            
            # Weather features
            features.append(self.weather_factors[weather_conditions])
            
            # Qualifying features
            features.extend([
                np.mean(qualifying_positions[:5]),  # Average top 5 qualifying
                np.std(qualifying_positions[:5]),   # Consistency of top 5
                np.mean(qualifying_positions[5:10]), # Average midfield qualifying
                np.std(qualifying_positions[5:10]),  # Consistency of midfield
                np.mean(qualifying_positions[10:]),  # Average backmarker qualifying
                np.std(qualifying_positions[10:]),   # Consistency of backmarkers
            ])
            
            # Practice features (if provided)
            if practice_results:
                if len(practice_results) != 3:
                    raise ValueError("Must provide exactly 3 practice sessions")
                
                for session in practice_results:
                    if len(session) != 20:
                        raise ValueError("Each practice session must have 20 drivers")
                    
                    features.extend([
                        np.mean(session),      # Average session time
                        np.std(session),       # Session consistency
                        np.min(session),       # Best time
                        np.max(session),       # Worst time
                    ])
            else:
                # Use default practice features
                features.extend([0] * 12)  # 3 sessions * 4 features each
            
            # Additional engineered features
            features.extend([
                np.mean(qualifying_positions),  # Overall qualifying average
                np.std(qualifying_positions),   # Overall qualifying spread
                max(qualifying_positions) - min(qualifying_positions),  # Qualifying range
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing prediction input: {e}")
            raise
    
    def validate_input_data(self, data: Dict) -> Tuple[bool, str]:
        """Validate input data for predictions."""
        try:
            required_fields = ['track', 'qualifying_positions']
            
            for field in required_fields:
                if field not in data:
                    return False, f"Missing required field: {field}"
            
            # Validate track
            if data['track'] not in self.track_features:
                return False, f"Unknown track: {data['track']}"
            
            # Validate qualifying positions
            qualifying = data['qualifying_positions']
            if not isinstance(qualifying, list) or len(qualifying) != 20:
                return False, "Qualifying positions must be a list of 20 integers"
            
            if not all(isinstance(pos, int) and 1 <= pos <= 20 for pos in qualifying):
                return False, "Qualifying positions must be integers between 1 and 20"
            
            # Validate practice results if provided
            if 'practice_results' in data:
                practice = data['practice_results']
                if not isinstance(practice, list):
                    return False, "Practice results must be a list"
                
                for session in practice:
                    if not isinstance(session, list) or len(session) != 20:
                        return False, "Each practice session must be a list of 20 times"
                    
                    if not all(isinstance(time, (int, float)) and time > 0 for time in session):
                        return False, "Practice times must be positive numbers"
            
            # Validate weather conditions
            if 'weather_conditions' in data:
                weather = data['weather_conditions']
                if weather not in self.weather_factors:
                    return False, f"Unknown weather condition: {weather}"
            
            return True, "Data validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str):
        """Save processed data to file."""
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Saved processed data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            'street_circuit', 'permanent_circuit', 'avg_speed_norm', 'corners_norm',
            'weather_factor', 'qual_top5_avg', 'qual_top5_std', 'qual_mid_avg', 'qual_mid_std',
            'qual_back_avg', 'qual_back_std', 'fp1_avg', 'fp1_std', 'fp1_best', 'fp1_worst',
            'fp2_avg', 'fp2_std', 'fp2_best', 'fp2_worst', 'fp3_avg', 'fp3_std', 'fp3_best', 'fp3_worst',
            'qual_overall_avg', 'qual_overall_std', 'qual_range'
        ] 