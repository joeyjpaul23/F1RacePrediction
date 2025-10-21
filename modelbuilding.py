import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your data
df = pd.read_csv('data/f1_results_2020.csv')  # Change as needed

# Only use features available before the race
features = ['Qual_Position', 'Race_GridPosition', 'Race_TeamName', 'Race_FullName']
df = df[features + ['Race_Position', 'EventName', 'Year', 'Round']]

# Drop rows with missing data
df = df.dropna()

# Encode categorical variables
df['Race_TeamName'] = df['Race_TeamName'].astype('category').cat.codes
df['Race_FullName'] = df['Race_FullName'].astype('category').cat.codes

# Create the label: 1 if this driver won the race, else 0
df['Winner'] = (df['Race_Position'] == 1).astype(int)

# Split into train and test sets by race (so the model never sees the winner for a race it's predicting)
unique_races = df[['Year', 'Round']].drop_duplicates()
train_races, test_races = train_test_split(unique_races, test_size=0.2, random_state=42)

train_idx = df.set_index(['Year', 'Round']).index.isin([tuple(x) for x in train_races.values])
test_idx = ~train_idx

train_df = df[train_idx]
test_df = df[test_idx]

X_train = train_df[features]
y_train = train_df['Winner']

X_test = test_df[features]
y_test = test_df['Winner']

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict the winner for each race in the test set
predicted_winners = []

for (year, rnd), group in test_df.groupby(['Year', 'Round']):
    X_group = group[features]
    probs = clf.predict_proba(X_group)[:, 1]  # Probability of being the winner
    idx_max = probs.argmax()
    predicted_driver = group.iloc[idx_max]['Race_FullName']
    actual_driver = group[group['Winner'] == 1]['Race_FullName'].values[0]
    event_name = group.iloc[0]['EventName']
    predicted_winners.append({
        'Year': year,
        'Round': rnd,
        'EventName': event_name,
        'PredictedWinner': predicted_driver,
        'ActualWinner': actual_driver
    })

# Show predictions
results_df = pd.DataFrame(predicted_winners)
print(results_df)

# Optional: Calculate how many times the model got it right
accuracy = (results_df['PredictedWinner'] == results_df['ActualWinner']).mean()
print(f'Prediction accuracy: {accuracy:.2%}')