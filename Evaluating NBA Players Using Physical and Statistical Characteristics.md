## Visualizing High-Dimensional Data with UMAP in Python
Slide 1: Introduction to NBA Player Evaluation

Evaluating NBA players using physical and statistical characteristics is a crucial aspect of team management and player scouting. This process involves analyzing various metrics to gain insights into a player's performance and potential. In this presentation, we'll explore how to use Python to collect, process, and visualize NBA player data.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load NBA player data
nba_data = pd.read_csv('nba_player_data.csv')

# Display the first few rows and basic information
print(nba_data.head())
print(nba_data.info())
```

Slide 2: Data Collection

The first step in evaluating NBA players is gathering relevant data. We'll use the NBA API to fetch player statistics and physical attributes.

```python
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players

# Get player ID
player_dict = players.get_players()
player_id = [player for player in player_dict if player['full_name'] == 'LeBron James'][0]['id']

# Fetch career stats
career = playercareerstats.PlayerCareerStats(player_id=player_id)
career_df = career.get_data_frames()[0]

print(career_df.head())
```

Slide 3: Physical Characteristics Analysis

Physical attributes like height, weight, and wingspan can provide insights into a player's potential performance in different positions.

```python
import seaborn as sns

# Assuming we have a DataFrame 'nba_data' with physical attributes
plt.figure(figsize=(10, 6))
sns.scatterplot(data=nba_data, x='Height', y='Weight', hue='Position')
plt.title('NBA Players: Height vs Weight by Position')
plt.show()

# Calculate average physical attributes by position
position_averages = nba_data.groupby('Position')[['Height', 'Weight', 'Wingspan']].mean()
print(position_averages)
```

Slide 4: Statistical Performance Metrics

Key statistical metrics like points per game (PPG), rebounds per game (RPG), and assists per game (APG) are essential for evaluating player performance.

```python
# Calculate basic performance metrics
nba_data['PPG'] = nba_data['Points'] / nba_data['Games Played']
nba_data['RPG'] = nba_data['Total Rebounds'] / nba_data['Games Played']
nba_data['APG'] = nba_data['Assists'] / nba_data['Games Played']

# Display top players by PPG
top_scorers = nba_data.sort_values('PPG', ascending=False).head(10)
print(top_scorers[['Player', 'PPG']])
```

Slide 5: Advanced Statistics

Advanced statistics like Player Efficiency Rating (PER) and True Shooting Percentage (TS%) provide a more comprehensive view of a player's impact.

```python
# Calculate PER (simplified version)
nba_data['PER'] = (nba_data['Points'] + nba_data['Total Rebounds'] + nba_data['Assists'] + 
                   nba_data['Steals'] + nba_data['Blocks'] - nba_data['Turnovers']) / nba_data['Minutes Played']

# Calculate TS%
nba_data['TS%'] = nba_data['Points'] / (2 * (nba_data['Field Goal Attempts'] + 0.44 * nba_data['Free Throw Attempts']))

# Display top players by PER
top_per = nba_data.sort_values('PER', ascending=False).head(10)
print(top_per[['Player', 'PER', 'TS%']])
```

Slide 6: Visualizing Player Performance

Creating visualizations can help in comparing players and identifying trends in performance.

```python
import matplotlib.pyplot as plt

# Create a scatter plot of PER vs TS%
plt.figure(figsize=(12, 8))
plt.scatter(nba_data['TS%'], nba_data['PER'], alpha=0.6)
plt.xlabel('True Shooting Percentage')
plt.ylabel('Player Efficiency Rating')
plt.title('PER vs TS% for NBA Players')

# Annotate some top players
for i, player in top_per.iterrows():
    plt.annotate(player['Player'], (player['TS%'], player['PER']))

plt.show()
```

Slide 7: Player Comparison

Comparing players across multiple metrics can provide a comprehensive view of their relative strengths and weaknesses.

```python
def radar_chart(players, stats):
    num_vars = len(stats)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for player in players:
        values = nba_data[nba_data['Player'] == player][stats].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=player)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), stats)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.show()

players_to_compare = ['LeBron James', 'Kevin Durant', 'Stephen Curry']
stats_to_compare = ['PPG', 'RPG', 'APG', 'TS%', 'PER']

radar_chart(players_to_compare, stats_to_compare)
```

Slide 8: Positional Analysis

Analyzing players within their specific positions can provide context for their performance and value to a team.

```python
# Calculate z-scores for key metrics within each position
metrics = ['PPG', 'RPG', 'APG', 'TS%', 'PER']

for metric in metrics:
    nba_data[f'{metric}_zscore'] = nba_data.groupby('Position')[metric].transform(lambda x: (x - x.mean()) / x.std())

# Find top players by position based on overall z-score
nba_data['overall_zscore'] = nba_data[[f'{metric}_zscore' for metric in metrics]].mean(axis=1)
top_by_position = nba_data.groupby('Position').apply(lambda x: x.nlargest(5, 'overall_zscore'))

print(top_by_position[['Player', 'Position', 'overall_zscore']])
```

Slide 9: Time Series Analysis

Analyzing a player's performance over time can reveal trends, improvements, or declines in their game.

```python
import matplotlib.dates as mdates

# Assuming we have career data for a specific player
player_career = career_df[career_df['PLAYER_ID'] == player_id]

plt.figure(figsize=(12, 6))
plt.plot(player_career['SEASON_ID'], player_career['PTS'], marker='o')
plt.title("LeBron James - Points per Season")
plt.xlabel("Season")
plt.ylabel("Points")
plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
plt.grid(True)
plt.show()
```

Slide 10: Clustering Players

Using machine learning techniques like K-means clustering can help identify similar player types based on their statistics.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features for clustering
features = ['PPG', 'RPG', 'APG', 'TS%', 'PER']

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(nba_data[features])

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
nba_data['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(data=nba_data, x='PPG', y='PER', hue='Cluster', palette='deep')
plt.title('NBA Player Clusters based on Performance Metrics')
plt.show()

# Display cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_df = pd.DataFrame(cluster_centers, columns=features)
print(cluster_df)
```

Slide 11: Predictive Modeling

Building predictive models can help estimate future performance or identify undervalued players.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Predict PER based on other stats
X = nba_data[['PPG', 'RPG', 'APG', 'TS%']]
y = nba_data['PER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print("R-squared score: ", r2_score(y_test, y_pred))

# Display feature importance
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
```

Slide 12: Real-Life Example: Draft Analysis

Evaluating college players for the NBA draft using physical and statistical characteristics.

```python
# Assume we have a DataFrame 'draft_prospects' with college stats and physical measurements

# Normalize stats to per-40-minute averages
per_40_stats = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks']
for stat in per_40_stats:
    draft_prospects[f'{stat}_per_40'] = draft_prospects[stat] / draft_prospects['Minutes Played'] * 40

# Create a composite score
draft_prospects['Draft_Score'] = (
    draft_prospects['Points_per_40'] * 0.3 +
    draft_prospects['Rebounds_per_40'] * 0.2 +
    draft_prospects['Assists_per_40'] * 0.2 +
    draft_prospects['Steals_per_40'] * 0.15 +
    draft_prospects['Blocks_per_40'] * 0.15
) * (draft_prospects['Height'] / draft_prospects['Height'].mean())

# Display top prospects
top_prospects = draft_prospects.sort_values('Draft_Score', ascending=False).head(10)
print(top_prospects[['Player', 'Draft_Score', 'Height', 'Points_per_40', 'Rebounds_per_40', 'Assists_per_40']])
```

Slide 13: Real-Life Example: Injury Risk Assessment

Using physical characteristics and playing time to assess injury risk for NBA players.

```python
import numpy as np

# Assume we have a DataFrame 'player_data' with relevant information

# Calculate a simple injury risk score
player_data['Injury_Risk_Score'] = (
    (player_data['Age'] * 0.2) +
    (player_data['Minutes_Played_Last_Season'] * 0.3) +
    (player_data['Previous_Injuries'] * 0.3) +
    (np.abs(player_data['BMI'] - 23) * 0.2)  # Assuming ideal BMI is around 23
)

# Normalize the score
player_data['Injury_Risk_Score'] = (player_data['Injury_Risk_Score'] - player_data['Injury_Risk_Score'].min()) / (player_data['Injury_Risk_Score'].max() - player_data['Injury_Risk_Score'].min())

# Display players with highest injury risk
high_risk_players = player_data.sort_values('Injury_Risk_Score', ascending=False).head(10)
print(high_risk_players[['Player', 'Age', 'Minutes_Played_Last_Season', 'Previous_Injuries', 'BMI', 'Injury_Risk_Score']])

# Visualize injury risk distribution
plt.figure(figsize=(10, 6))
sns.histplot(player_data['Injury_Risk_Score'], kde=True)
plt.title('Distribution of Injury Risk Scores')
plt.xlabel('Injury Risk Score')
plt.ylabel('Count')
plt.show()
```

Slide 14: Conclusion

Evaluating NBA players using physical and statistical characteristics is a complex process that requires a combination of data analysis, statistical modeling, and domain knowledge. Python provides powerful tools for collecting, processing, and visualizing NBA player data, enabling teams and analysts to make data-driven decisions in player evaluation, team building, and strategy development.

Slide 15: Additional Resources

For those interested in diving deeper into NBA analytics and player evaluation, here are some valuable resources:

1. "Predicting NBA Player Performance" by Terran Gilmore et al. (2018) ArXiv: [https://arxiv.org/abs/1809.05118](https://arxiv.org/abs/1809.05118)
2. "A Network-Based Analysis of Basketball Team Strategies" by Javier López Peña and Hugo Touchette (2012) ArXiv: [https://arxiv.org/abs/1206.7004](https://arxiv.org/abs/1206.7004)
3. "The Science of the NBA Basketball Free Throw" by Matthew J. Berntsen (2016) ArXiv: [https://arxiv.org/abs/1701.01616](https://arxiv.org/abs/1701.01616)

These papers provide in-depth analyses and methodologies for evaluating basketball players and team strategies using advanced statistical and mathematical techniques.

