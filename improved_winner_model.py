import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load F1 data"""
    print("Loading F1 data...")
    
    import os
    data_folder = 'data'
    files = os.listdir(data_folder)
    
    dfs = []
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_folder, file))
            dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.sort_values(by=['Year', 'Round'], ascending=True, inplace=True)
    
    print(f"Loaded {len(combined_df)} records from {len(files)} files")
    return combined_df

def create_advanced_features(df):
    """Create advanced features for better prediction"""
    print("Creating advanced features...")
    
    df_enhanced = df.copy()
    
    # Handle missing values
    df_enhanced['Race_Position'] = pd.to_numeric(df_enhanced['Race_Position'], errors='coerce')
    df_enhanced['Qual_Position'] = pd.to_numeric(df_enhanced['Qual_Position'], errors='coerce')
    
    # Remove rows where target variable is missing
    df_enhanced = df_enhanced.dropna(subset=['Race_Position'])
    
    # Create winner target
    df_enhanced['Winner'] = (df_enhanced['Race_Position'] == 1).astype(int)
    
    # Convert qualifying times to seconds
    for col in ['Qual_Q1', 'Qual_Q2', 'Qual_Q3']:
        df_enhanced[col] = pd.to_timedelta(df_enhanced[col], errors='coerce').dt.total_seconds()
    
    # Create qualifying performance features
    df_enhanced['Qual_Avg'] = df_enhanced[['Qual_Q1', 'Qual_Q2', 'Qual_Q3']].mean(axis=1)
    df_enhanced['Qual_Best'] = df_enhanced[['Qual_Q1', 'Qual_Q2', 'Qual_Q3']].min(axis=1)
    df_enhanced['Qual_Consistency'] = df_enhanced[['Qual_Q1', 'Qual_Q2', 'Qual_Q3']].std(axis=1)
    
    # Create driver experience features
    df_enhanced['Driver_Experience'] = df_enhanced.groupby('Race_DriverId')['Year'].rank(method='dense')
    
    # Create rolling performance features (last 5 races)
    df_enhanced = df_enhanced.sort_values(['Race_DriverId', 'Year', 'Round'])
    
    # Driver's average position in last 5 races
    df_enhanced['Driver_Last5_Avg_Position'] = df_enhanced.groupby('Race_DriverId')['Race_Position'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    
    # Driver's best position in last 5 races
    df_enhanced['Driver_Last5_Best_Position'] = df_enhanced.groupby('Race_DriverId')['Race_Position'].rolling(5, min_periods=1).min().reset_index(0, drop=True)
    
    # Driver's qualifying performance in last 5 races
    df_enhanced['Driver_Last5_Avg_Qual'] = df_enhanced.groupby('Race_DriverId')['Qual_Position'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    
    # Driver's win rate in last 10 races
    df_enhanced['Driver_Last10_Win_Rate'] = df_enhanced.groupby('Race_DriverId')['Winner'].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
    
    # Team performance features
    df_enhanced = df_enhanced.sort_values(['Race_TeamId', 'Year', 'Round'])
    
    # Team's average position in last 5 races
    df_enhanced['Team_Last5_Avg_Position'] = df_enhanced.groupby('Race_TeamId')['Race_Position'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    
    # Team's best position in last 5 races
    df_enhanced['Team_Last5_Best_Position'] = df_enhanced.groupby('Race_TeamId')['Race_Position'].rolling(5, min_periods=1).min().reset_index(0, drop=True)
    
    # Team's win rate in last 10 races
    df_enhanced['Team_Last10_Win_Rate'] = df_enhanced.groupby('Race_TeamId')['Winner'].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
    
    # Season performance features
    season_performance = df_enhanced.groupby(['Race_DriverId', 'Year']).agg({
        'Race_Position': ['mean', 'min', 'std'],
        'Race_Points': 'sum',
        'Qual_Position': ['mean', 'min'],
        'Winner': 'sum'
    }).reset_index()
    
    season_performance.columns = ['Race_DriverId', 'Year', 'Season_Avg_Position', 'Season_Best_Position', 
                                 'Season_Position_Std', 'Season_Total_Points', 'Season_Avg_Qual', 'Season_Best_Qual', 'Season_Wins']
    
    df_enhanced = df_enhanced.merge(season_performance, on=['Race_DriverId', 'Year'], how='left')
    
    # Team season performance
    team_season_performance = df_enhanced.groupby(['Race_TeamId', 'Year']).agg({
        'Race_Position': ['mean', 'min'],
        'Race_Points': 'sum',
        'Winner': 'sum'
    }).reset_index()
    
    team_season_performance.columns = ['Race_TeamId', 'Year', 'Team_Season_Avg_Position', 'Team_Season_Best_Position', 'Team_Season_Total_Points', 'Team_Season_Wins']
    
    df_enhanced = df_enhanced.merge(team_season_performance, on=['Race_TeamId', 'Year'], how='left')
    
    # Circuit-specific performance
    circuit_performance = df_enhanced.groupby(['Race_DriverId', 'EventName']).agg({
        'Race_Position': ['mean', 'min', 'count'],
        'Winner': 'sum'
    }).reset_index()
    
    circuit_performance.columns = ['Race_DriverId', 'EventName', 'Circuit_Avg_Position', 'Circuit_Best_Position', 'Circuit_Races', 'Circuit_Wins']
    
    df_enhanced = df_enhanced.merge(circuit_performance, on=['Race_DriverId', 'EventName'], how='left')
    
    # Fill missing values
    df_enhanced['Circuit_Races'] = df_enhanced['Circuit_Races'].fillna(0)
    df_enhanced['Circuit_Wins'] = df_enhanced['Circuit_Wins'].fillna(0)
    df_enhanced['Circuit_Avg_Position'] = df_enhanced['Circuit_Avg_Position'].fillna(df_enhanced['Season_Avg_Position'])
    df_enhanced['Circuit_Best_Position'] = df_enhanced['Circuit_Best_Position'].fillna(df_enhanced['Season_Best_Position'])
    
    # Create additional features
    df_enhanced['Driver_Win_Rate'] = df_enhanced['Season_Wins'] / df_enhanced.groupby(['Race_DriverId', 'Year'])['Race_Position'].transform('count')
    df_enhanced['Team_Win_Rate'] = df_enhanced['Team_Season_Wins'] / df_enhanced.groupby(['Race_TeamId', 'Year'])['Race_Position'].transform('count')
    df_enhanced['Circuit_Win_Rate'] = df_enhanced['Circuit_Wins'] / df_enhanced['Circuit_Races']
    
    # Fill NaN win rates
    df_enhanced['Driver_Win_Rate'] = df_enhanced['Driver_Win_Rate'].fillna(0)
    df_enhanced['Team_Win_Rate'] = df_enhanced['Team_Win_Rate'].fillna(0)
    df_enhanced['Circuit_Win_Rate'] = df_enhanced['Circuit_Win_Rate'].fillna(0)
    
    print(f"Created {len(df_enhanced.columns)} features")
    return df_enhanced

def prepare_modeling_data(df):
    """Prepare data for modeling"""
    print("Preparing modeling data...")
    
    # Select features for modeling
    feature_columns = [
        'Qual_Position',
        'Qual_Avg',
        'Qual_Best', 
        'Qual_Consistency',
        'Driver_Experience',
        'Driver_Last5_Avg_Position',
        'Driver_Last5_Best_Position',
        'Driver_Last5_Avg_Qual',
        'Driver_Last10_Win_Rate',
        'Team_Last5_Avg_Position',
        'Team_Last5_Best_Position',
        'Team_Last10_Win_Rate',
        'Season_Avg_Position',
        'Season_Best_Position',
        'Season_Position_Std',
        'Season_Total_Points',
        'Season_Avg_Qual',
        'Season_Best_Qual',
        'Season_Wins',
        'Driver_Win_Rate',
        'Team_Season_Avg_Position',
        'Team_Season_Best_Position',
        'Team_Season_Total_Points',
        'Team_Season_Wins',
        'Team_Win_Rate',
        'Circuit_Avg_Position',
        'Circuit_Best_Position',
        'Circuit_Races',
        'Circuit_Wins',
        'Circuit_Win_Rate',
        'Year',
        'Round'
    ]
    
    # Encode categorical variables
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    
    df['Driver_Encoded'] = le_driver.fit_transform(df['Race_DriverId'])
    df['Team_Encoded'] = le_team.fit_transform(df['Race_TeamId'])
    
    feature_columns.extend(['Driver_Encoded', 'Team_Encoded'])
    
    # Prepare X and y
    X = df[feature_columns].copy()
    y = df['Winner']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Remove any remaining NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"Final dataset: {len(X)} samples, {len(feature_columns)} features")
    print(f"Winner rate: {y.mean():.2%}")
    
    return X, y, feature_columns

def train_ensemble_model(X, y):
    """Train an ensemble model with class weights"""
    print("Training ensemble model...")
    
    # Split data by time
    df_temp = pd.DataFrame({'X': X.values.tolist(), 'y': y.values})
    df_temp['Year'] = X['Year'].values
    df_temp['Round'] = X['Round'].values
    
    # Use 80% of races for training, 20% for testing
    unique_races = df_temp[['Year', 'Round']].drop_duplicates().sort_values(['Year', 'Round'])
    split_idx = int(len(unique_races) * 0.8)
    train_races = unique_races.iloc[:split_idx]
    test_races = unique_races.iloc[split_idx:]
    
    train_mask = df_temp.set_index(['Year', 'Round']).index.isin([tuple(x) for x in train_races.values])
    test_mask = df_temp.set_index(['Year', 'Round']).index.isin([tuple(x) for x in test_races.values])
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute class weights to handle imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    print(f"Class weights: {class_weight_dict}")
    
    # Create ensemble model
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight=class_weight_dict, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
    lr = LogisticRegression(class_weight=class_weight_dict, max_iter=1000, random_state=42)
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft'
    )
    
    # Train ensemble
    ensemble.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = ensemble.predict(X_test_scaled)
    y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble model accuracy: {accuracy:.3f}")
    
    return ensemble, scaler, X_test, y_test, y_pred, y_pred_proba, X_train_scaled, y_train

def analyze_predictions(df_test, y_pred, y_pred_proba, y_test):
    """Analyze and display predictions"""
    print("\n" + "="*80)
    print("IMPROVED WINNER PREDICTION RESULTS")
    print("="*80)
    
    # Add predictions to test data
    df_test['Predicted_Winner'] = y_pred
    df_test['Win_Probability'] = y_pred_proba
    
    # Group by race and analyze
    race_results = []
    
    for (year, round_num), race_group in df_test.groupby(['Year', 'Round']):
        # Sort by win probability
        race_group_sorted = race_group.sort_values('Win_Probability', ascending=False)
        
        # Get actual winner and predicted winner
        actual_winner = race_group[race_group['Winner'] == 1]['Race_FullName'].iloc[0]
        predicted_winner = race_group_sorted.iloc[0]['Race_FullName']
        predicted_prob = race_group_sorted.iloc[0]['Win_Probability']
        
        # Get top 3 predictions
        top3 = race_group_sorted.head(3)[['Race_FullName', 'Win_Probability']]
        
        winner_correct = actual_winner == predicted_winner
        
        race_results.append({
            'Year': year,
            'Round': round_num,
            'EventName': race_group['EventName'].iloc[0],
            'Actual_Winner': actual_winner,
            'Predicted_Winner': predicted_winner,
            'Predicted_Probability': predicted_prob,
            'Winner_Correct': winner_correct,
            'Top3_Predictions': top3
        })
        
        # Print results for this race
        print(f"\n{year} Round {round_num}: {race_group['EventName'].iloc[0]}")
        print(f"  Actual Winner:    {actual_winner}")
        print(f"  Predicted Winner: {predicted_winner} (Prob: {predicted_prob:.3f})")
        print(f"  Correct:          {'✓' if winner_correct else '✗'}")
        print(f"  Top 3 Predictions:")
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            print(f"    {i}. {row['Race_FullName']} ({row['Win_Probability']:.3f})")
    
    # Calculate overall accuracy
    total_races = len(race_results)
    correct_predictions = sum(1 for race in race_results if race['Winner_Correct'])
    accuracy = correct_predictions / total_races
    
    print("\n" + "="*80)
    print(f"OVERALL RESULTS")
    print("="*80)
    print(f"Total Races: {total_races}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print("="*80)
    
    return race_results, accuracy

def main():
    """Main function"""
    print("Starting Improved F1 Winner Prediction Model")
    print("="*60)
    
    # Load and prepare data
    df = load_data()
    df_enhanced = create_advanced_features(df)
    X, y, feature_columns = prepare_modeling_data(df_enhanced)
    
    # Train ensemble model
    ensemble, scaler, X_test, y_test, y_pred, y_pred_proba, X_train_scaled, y_train = train_ensemble_model(X, y)
    
    # Get test data with full information
    df_test = df_enhanced[df_enhanced.set_index(['Year', 'Round']).index.isin(
        X_test.set_index(['Year', 'Round']).index
    )].copy()
    
    # Analyze predictions
    race_results, accuracy = analyze_predictions(df_test, y_pred, y_pred_proba, y_test)
    
    print(f"\nModel training completed!")
    print(f"Final accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main() 