## Simplify Data Science Workflow with CRISP-DM in Python
Slide 1: Introduction to CRISP-DM

CRISP-DM (Cross-Industry Standard Process for Data Mining) is a widely used methodology for data science projects. It provides a structured approach to planning and executing data science workflows, ensuring consistency and efficiency. This presentation will guide you through the CRISP-DM process using Python, demonstrating how to simplify your data science workflow.

```python
import matplotlib.pyplot as plt

crisp_dm_phases = ['Business Understanding', 'Data Understanding', 'Data Preparation', 
                   'Modeling', 'Evaluation', 'Deployment']

plt.figure(figsize=(10, 6))
plt.pie([1]*6, labels=crisp_dm_phases, autopct='%1.1f%%')
plt.title('CRISP-DM Phases')
plt.axis('equal')
plt.show()
```

Slide 2: Business Understanding

The first phase of CRISP-DM focuses on understanding the project objectives from a business perspective. This involves defining the problem, identifying key stakeholders, and setting project goals. In Python, we can use libraries like pandas to analyze business metrics and visualize trends.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        'Sales': [100, 120, 140, 130, 150],
        'Customer_Satisfaction': [4.2, 4.3, 4.1, 4.4, 4.5]}

df = pd.DataFrame(data)

# Visualize sales trend
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Sales'], marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

# Calculate correlation between sales and customer satisfaction
correlation = df['Sales'].corr(df['Customer_Satisfaction'])
print(f"Correlation between Sales and Customer Satisfaction: {correlation:.2f}")
```

Slide 3: Data Understanding

In this phase, we collect and explore the initial data to identify data quality issues and gain insights. Python's pandas library is excellent for data exploration and basic statistical analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load a sample dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Display basic information about the dataset
print(df.info())

# Show summary statistics
print(df.describe())

# Visualize the distribution of a numerical column
plt.figure(figsize=(10, 6))
df['Age'].hist(bins=20)
plt.title('Distribution of Passenger Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])
```

Slide 4: Data Preparation

Data preparation involves cleaning, transforming, and formatting the data for modeling. This often includes handling missing values, encoding categorical variables, and scaling numerical features.

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Scale numerical features
scaler = StandardScaler()
df['Fare'] = scaler.fit_transform(df[['Fare']])

# Drop unnecessary columns
df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

print(df.head())
print("\nDataset shape after preparation:", df.shape)
```

Slide 5: Modeling

The modeling phase involves selecting and applying various machine learning algorithms to the prepared data. We'll use scikit-learn to build and train a simple model.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Prepare features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

Slide 6: Evaluation

In the evaluation phase, we assess the model's performance against the business objectives. This often involves using various metrics and techniques to ensure the model meets the project goals.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get probability predictions
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("Top 5 important features:")
print(feature_importance.head())
```

Slide 7: Deployment

The deployment phase involves integrating the model into the production environment. We'll create a simple function to make predictions using our trained model.

```python
import joblib

# Save the model
joblib.dump(rf_model, 'titanic_survival_model.joblib')

# Function to make predictions
def predict_survival(passenger_data):
    # Load the model
    loaded_model = joblib.load('titanic_survival_model.joblib')
    
    # Make prediction
    prediction = loaded_model.predict(passenger_data)
    probability = loaded_model.predict_proba(passenger_data)[:, 1]
    
    return prediction[0], probability[0]

# Example usage
new_passenger = [[3, 0, 22.0, 1, 0, 7.25]]  # [Pclass, Sex, Age, SibSp, Parch, Fare]
prediction, probability = predict_survival(new_passenger)

print(f"Survival Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")
print(f"Probability of Survival: {probability:.2f}")
```

Slide 8: Real-life Example: Customer Churn Prediction

Let's apply CRISP-DM to a real-world problem: predicting customer churn for a telecom company. We'll go through each phase of the process.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset (you would typically load your own data here)
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Data Understanding and Preparation
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Convert categorical variables to numeric
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
categorical_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Prepare features and target
X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
```

Slide 9: Real-life Example: Customer Churn Prediction (Continued)

Let's visualize the results and interpret the model for our customer churn prediction example.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 10 Important Features for Churn Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Churn Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Calculate and print churn rate
churn_rate = y.mean()
print(f"Overall Churn Rate: {churn_rate:.2%}")

# Interpret results
print("\nModel Interpretation:")
print("1. The model achieved good performance in predicting customer churn.")
print("2. Key factors influencing churn include contract type, tenure, and monthly charges.")
print("3. The confusion matrix shows the model's ability to correctly identify churners and non-churners.")
print("4. This information can be used to develop targeted retention strategies for high-risk customers.")
```

Slide 10: Iterative Nature of CRISP-DM

CRISP-DM is an iterative process. After completing one cycle, we often return to earlier phases to refine our approach based on insights gained. Let's simulate this by improving our churn prediction model.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}

# Perform randomized search
rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_distributions=param_grid,
                               n_iter=100,
                               cv=5,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X_train_scaled, y_train)

# Print the best parameters
print("Best parameters found:")
print(rf_random.best_params_)

# Evaluate the improved model
y_pred_improved = rf_random.predict(X_test_scaled)
print("\nImproved Model Classification Report:")
print(classification_report(y_test, y_pred_improved))

# Compare ROC curves
from sklearn.metrics import roc_curve, auc

# Original model
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Improved model
y_pred_proba_improved = rf_random.predict_proba(X_test_scaled)[:, 1]
fpr_improved, tpr_improved, _ = roc_curve(y_test, y_pred_proba_improved)
roc_auc_improved = auc(fpr_improved, tpr_improved)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Original Model (AUC = {roc_auc:.2f})')
plt.plot(fpr_improved, tpr_improved, color='red', lw=2, label=f'Improved Model (AUC = {roc_auc_improved:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
plt.legend(loc="lower right")
plt.show()
```

Slide 11: Automating CRISP-DM Workflow

To further simplify the data science workflow, we can create a Python class that encapsulates the CRISP-DM process. This allows for easier replication and modification of the workflow for different projects.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

class CRISP_DM_Workflow:
    def __init__(self, data_path, target_column):
        self.data_path = data_path
        self.target_column = target_column
        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.model = None
    
    def business_understanding(self):
        print("Phase 1: Business Understanding")
        print("Objective: Predict target variable to improve business strategies.")
    
    def data_understanding(self):
        print("\nPhase 2: Data Understanding")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {', '.join(self.df.columns)}")
        print(f"Missing values:\n{self.df.isnull().sum()}")
    
    def data_preparation(self):
        print("\nPhase 3: Data Preparation")
        self.df = self.df.dropna()
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        
    def modeling(self):
        print("\nPhase 4: Modeling")
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)
    
    def evaluation(self):
        print("\nPhase 5: Evaluation")
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
    
    def deployment(self):
        print("\nPhase 6: Deployment")
        print("Model ready for deployment in production environment.")

# Usage
workflow = CRISP_DM_Workflow('data.csv', 'target')
workflow.business_understanding()
workflow.data_understanding()
workflow.data_preparation()
workflow.modeling()
workflow.evaluation()
workflow.deployment()
```

Slide 12: Handling Imbalanced Datasets

In many real-world scenarios, datasets can be imbalanced, which may affect model performance. Let's explore techniques to handle this issue within the CRISP-DM framework.

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Assume we have imbalanced X_train and y_train

# Create a pipeline with SMOTE oversampling and Random Forest
imbalanced_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the pipeline
imbalanced_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_balanced = imbalanced_pipeline.predict(X_test)

# Evaluate the model
print("Classification Report (Balanced):")
print(classification_report(y_test, y_pred_balanced))

# Compare with original model
y_pred_original = RandomForestClassifier(random_state=42).fit(X_train, y_train).predict(X_test)
print("\nClassification Report (Original):")
print(classification_report(y_test, y_pred_original))

# Visualize class distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
y_train.value_counts().plot(kind='bar')
plt.title('Original Class Distribution')
plt.subplot(1, 2, 2)
pd.Series(imbalanced_pipeline.named_steps['smote'].fit_resample(X_train, y_train)[1]).value_counts().plot(kind='bar')
plt.title('Balanced Class Distribution')
plt.tight_layout()
plt.show()
```

Slide 13: Feature Engineering in CRISP-DM

Feature engineering is a crucial step in the data preparation phase of CRISP-DM. Let's explore some common feature engineering techniques using Python.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer

# Assume we have a DataFrame 'df' with features

# 1. Creating interaction terms
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['feature1', 'feature2']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names(['feature1', 'feature2']))

# 2. Binning continuous variables
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '65+'])

# 3. Creating dummy variables
dummy_df = pd.get_dummies(df['categorical_column'], prefix='category')

# 4. Handling missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 5. Scaling features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

# 6. Creating date-based features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

print("Original DataFrame shape:", df.shape)
print("DataFrame shape after feature engineering:", df_scaled.shape)
```

Slide 14: Model Interpretability in CRISP-DM

Model interpretability is crucial for gaining insights and building trust in your models. Let's explore some techniques for interpreting our models within the CRISP-DM framework.

```python
import shap
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Assume we have X_train, X_test, y_train, y_test

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.title('Top 10 Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# SHAP values for model interpretation
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Individual prediction explanation
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0])

print("Model interpretation helps understand the factors influencing predictions.")
print("This aids in refining the model and gaining business insights.")
```

Slide 15: Additional Resources

For further exploration of CRISP-DM and data science workflows using Python, consider the following resources:

1. "Mastering the CRISP-DM Methodology: A Comprehensive Guide to Data Mining Projects" by M. Brown et al. (2020), arXiv:2006.10455 \[cs.LG\] URL: [https://arxiv.org/abs/2006.10455](https://arxiv.org/abs/2006.10455)
2. "A Survey of Data Mining and Machine Learning Methods for Cyber Security Intrusion Detection" by A. L. Buczak and E. Guven (2016), IEEE Communications Surveys & Tutorials URL: [https://ieeexplore.ieee.org/document/7307098](https://ieeexplore.ieee.org/document/7307098)
3. "Automated Machine Learning: Methods, Systems, Challenges" edited by F. Hutter et al. (2019), Springer ISBN: 978-3-030-05318-5
4. Python Data Science Handbook by Jake VanderPlas (2016), O'Reilly Media ISBN: 978-1-491-91205-8
5. Scikit-learn Documentation: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)

These resources provide in-depth knowledge on CRISP-DM, machine learning techniques, and Python libraries for data science. They can help you further refine your data science workflow and expand your skillset.

