## Predicting Olympic Outcomes with Python and Machine Learning
Slide 1: Introduction to Olympic Outcome Prediction

Predicting Olympic outcomes using machine learning and Python is an exciting application of data science in sports. This process involves collecting historical data, preprocessing it, training a model, and making predictions for future events. Let's explore how to build a predictive model for Olympic performances.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)
```

Slide 2: Data Collection

The first step in predicting Olympic outcomes is gathering relevant data. This includes historical Olympic results, athlete performance metrics, and other factors that might influence outcomes. Let's create a sample dataset to work with.

```python
# Create a sample dataset
data = {
    'athlete_id': range(1, 101),
    'age': np.random.randint(18, 35, 100),
    'experience': np.random.randint(1, 15, 100),
    'previous_medals': np.random.randint(0, 5, 100),
    'world_ranking': np.random.randint(1, 101, 100),
    'performance_score': np.random.uniform(70, 100, 100)
}

df = pd.DataFrame(data)
print(df.head())
```

Slide 3: Data Preprocessing

Before we can use our data for machine learning, we need to preprocess it. This involves handling missing values, encoding categorical variables, and scaling numerical features. In our example, we'll focus on scaling.

```python
# Select features and target
X = df[['age', 'experience', 'previous_medals', 'world_ranking']]
y = df['performance_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Shape of training data:", X_train_scaled.shape)
print("Shape of testing data:", X_test_scaled.shape)
```

Slide 4: Model Selection and Training

For this example, we'll use a Random Forest Regressor to predict performance scores. This model works well with numerical data and can capture complex relationships between features.

```python
# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

Slide 5: Feature Importance

Understanding which features have the most impact on our predictions can provide valuable insights. Random Forest models allow us to easily extract feature importance.

```python
# Get feature importance
importance = model.feature_importance_
feature_names = X.columns

# Sort features by importance
sorted_idx = importance.argsort()
sorted_features = feature_names[sorted_idx]
sorted_importance = importance[sorted_idx]

# Plot feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_importance)), sorted_importance)
plt.yticks(range(len(sorted_importance)), sorted_features)
plt.xlabel('Importance')
plt.title('Feature Importance in Olympic Performance Prediction')
plt.show()
```

Slide 6: Making Predictions for Future Olympics

Now that we have a trained model, we can use it to make predictions for future Olympic events. Let's create a function to predict an athlete's performance based on their characteristics.

```python
def predict_performance(age, experience, previous_medals, world_ranking):
    # Create a DataFrame with the athlete's data
    athlete_data = pd.DataFrame({
        'age': [age],
        'experience': [experience],
        'previous_medals': [previous_medals],
        'world_ranking': [world_ranking]
    })
    
    # Scale the data
    athlete_data_scaled = scaler.transform(athlete_data)
    
    # Make prediction
    prediction = model.predict(athlete_data_scaled)
    
    return prediction[0]

# Example prediction
athlete_performance = predict_performance(25, 8, 2, 5)
print(f"Predicted performance score: {athlete_performance:.2f}")
```

Slide 7: Identifying Potential Winners

To identify potential winners, we can predict performance scores for a group of athletes and rank them based on their predicted scores. Let's create a function to do this.

```python
def identify_potential_winners(athletes):
    predictions = []
    for athlete in athletes:
        score = predict_performance(*athlete)
        predictions.append((athlete, score))
    
    # Sort athletes by predicted score in descending order
    ranked_athletes = sorted(predictions, key=lambda x: x[1], reverse=True)
    return ranked_athletes

# Example usage
athletes = [
    (28, 10, 3, 2),  # (age, experience, previous_medals, world_ranking)
    (22, 4, 1, 15),
    (30, 12, 4, 1),
    (26, 7, 2, 8)
]

top_athletes = identify_potential_winners(athletes)
for i, (athlete, score) in enumerate(top_athletes, 1):
    print(f"Rank {i}: Athlete {athlete} - Predicted score: {score:.2f}")
```

Slide 8: Model Evaluation and Improvement

To ensure our model's predictions are reliable, we need to evaluate its performance and look for ways to improve it. Cross-validation is a powerful technique for this purpose.

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

# Convert MSE to RMSE
rmse_scores = np.sqrt(-cv_scores)

print(f"Cross-validation RMSE scores: {rmse_scores}")
print(f"Mean RMSE: {rmse_scores.mean():.2f}")
print(f"Standard deviation of RMSE: {rmse_scores.std():.2f}")
```

Slide 9: Handling Time Series Data

Olympic performances often have a time component. Let's explore how to incorporate time-based features into our model.

```python
import datetime

# Add a 'year' column to our dataset
df['year'] = np.random.randint(2000, 2021, len(df))

# Calculate 'years_since_first_olympics' feature
df['first_olympics'] = df['year'] - df['experience']
df['years_since_first_olympics'] = datetime.datetime.now().year - df['first_olympics']

# Update our feature set
X = df[['age', 'experience', 'previous_medals', 'world_ranking', 'years_since_first_olympics']]

print(df[['athlete_id', 'year', 'first_olympics', 'years_since_first_olympics']].head())
```

Slide 10: Handling Categorical Data

In real-world scenarios, we often encounter categorical data such as an athlete's country or sport. Let's see how to incorporate this information into our model.

```python
# Add categorical features
df['country'] = np.random.choice(['USA', 'China', 'Russia', 'Germany', 'Japan'], len(df))
df['sport'] = np.random.choice(['Swimming', 'Athletics', 'Gymnastics', 'Cycling', 'Rowing'], len(df))

# One-hot encode categorical variables
X = pd.get_dummies(df[['age', 'experience', 'previous_medals', 'world_ranking', 'years_since_first_olympics', 'country', 'sport']])

print(X.head())
```

Slide 11: Ensemble Methods for Improved Predictions

To further improve our predictions, we can use ensemble methods that combine multiple models. Let's implement a simple ensemble using Random Forest and Gradient Boosting.

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Create and train individual models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train_scaled, y_train)
gb_model.fit(X_train_scaled, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test_scaled)
gb_pred = gb_model.predict(X_test_scaled)

# Combine predictions (simple average)
ensemble_pred = (rf_pred + gb_pred) / 2

# Evaluate ensemble performance
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
print(f"Ensemble Model MAE: {ensemble_mae:.2f}")
```

Slide 12: Visualizing Predictions

Visualizing our predictions can help us understand the model's performance and identify any patterns or outliers.

```python
import seaborn as sns

# Create a DataFrame with actual and predicted values
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': ensemble_pred
})

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual', y='Predicted', data=results_df)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Performance Score')
plt.ylabel('Predicted Performance Score')
plt.title('Actual vs Predicted Olympic Performance Scores')
plt.show()
```

Slide 13: Real-life Example: Predicting Swimming Performance

Let's apply our model to predict swimming performance based on an athlete's 50m freestyle time and their average training hours per week.

```python
# Create a sample dataset for swimmers
swimmers_data = {
    'athlete_id': range(1, 51),
    '50m_freestyle_time': np.random.uniform(21.0, 25.0, 50),
    'avg_training_hours': np.random.uniform(20, 40, 50),
    'performance_score': np.random.uniform(70, 100, 50)
}

swimmers_df = pd.DataFrame(swimmers_data)

# Prepare the data
X_swim = swimmers_df[['50m_freestyle_time', 'avg_training_hours']]
y_swim = swimmers_df['performance_score']

# Split the data and train the model
X_train_swim, X_test_swim, y_train_swim, y_test_swim = train_test_split(X_swim, y_swim, test_size=0.2, random_state=42)
swim_model = RandomForestRegressor(n_estimators=100, random_state=42)
swim_model.fit(X_train_swim, y_train_swim)

# Make predictions
y_pred_swim = swim_model.predict(X_test_swim)

# Evaluate the model
swim_mse = mean_squared_error(y_test_swim, y_pred_swim)
print(f"Swimming Model MSE: {swim_mse:.2f}")

# Predict performance for a new swimmer
new_swimmer = [[22.5, 35]]  # 50m freestyle time: 22.5s, avg training hours: 35
predicted_score = swim_model.predict(new_swimmer)
print(f"Predicted performance score for new swimmer: {predicted_score[0]:.2f}")
```

Slide 14: Real-life Example: Predicting Track and Field Performance

Now, let's create a model to predict performance in track and field events, focusing on the 100m sprint and long jump.

```python
# Create a sample dataset for track and field athletes
track_field_data = {
    'athlete_id': range(1, 51),
    '100m_sprint_time': np.random.uniform(9.8, 11.0, 50),
    'long_jump_distance': np.random.uniform(7.5, 8.5, 50),
    'performance_score': np.random.uniform(70, 100, 50)
}

track_field_df = pd.DataFrame(track_field_data)

# Prepare the data
X_track = track_field_df[['100m_sprint_time', 'long_jump_distance']]
y_track = track_field_df['performance_score']

# Split the data and train the model
X_train_track, X_test_track, y_train_track, y_test_track = train_test_split(X_track, y_track, test_size=0.2, random_state=42)
track_model = RandomForestRegressor(n_estimators=100, random_state=42)
track_model.fit(X_train_track, y_train_track)

# Make predictions
y_pred_track = track_model.predict(X_test_track)

# Evaluate the model
track_mse = mean_squared_error(y_test_track, y_pred_track)
print(f"Track and Field Model MSE: {track_mse:.2f}")

# Predict performance for a new track and field athlete
new_athlete = [[10.2, 8.1]]  # 100m sprint time: 10.2s, long jump distance: 8.1m
predicted_score = track_model.predict(new_athlete)
print(f"Predicted performance score for new track and field athlete: {predicted_score[0]:.2f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into Olympic performance prediction and sports analytics, here are some valuable resources:

1. "Machine Learning for Sport Result Prediction" by Andrej Gajdoš et al. (2021) - Available on arXiv: [https://arxiv.org/abs/2101.03908](https://arxiv.org/abs/2101.03908)
2. "A Review of Machine Learning Applications in Olympic-Style Weightlifting" by Sergei Gepshtein et al. (2023) - Available on arXiv: [https://arxiv.org/abs/2303.10528](https://arxiv.org/abs/2303.10528)

These papers provide insights into advanced techniques and methodologies for predicting sports performance, including Olympic events.

