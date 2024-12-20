## Machine Learning Model Lifecycle! Scoping, Data, Modeling, Deployment
Slide 1: Machine Learning Model Lifecycle: Scoping

The scoping phase is crucial for defining project goals and requirements. It involves identifying the problem, determining feasibility, and setting success metrics.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define project goals
goals = ["Improve customer retention", "Increase sales", "Reduce churn"]

# Set success metrics
metrics = {
    "customer_retention": 0.85,
    "sales_increase": 0.15,
    "churn_reduction": 0.20
}

# Visualize goals and metrics
fig, ax = plt.subplots()
ax.bar(goals, list(metrics.values()))
ax.set_ylabel("Target Value")
ax.set_title("Project Goals and Success Metrics")
plt.show()
```

Slide 2: Defining the Problem Statement

A clear problem statement guides the entire machine learning project. It should be specific, measurable, and aligned with business objectives.

```python
def define_problem_statement(business_objective, target_variable, constraints):
    problem_statement = f"Develop a machine learning model to {business_objective} "
    problem_statement += f"by predicting {target_variable}, "
    problem_statement += f"subject to {constraints}."
    return problem_statement

business_objective = "improve customer retention"
target_variable = "likelihood of customer churn"
constraints = "maintaining data privacy and model interpretability"

problem_statement = define_problem_statement(business_objective, target_variable, constraints)
print(problem_statement)
```

Slide 3: Data Collection and Preprocessing

Data collection involves gathering relevant information from various sources. Preprocessing includes cleaning, handling missing values, and formatting the data for analysis.

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("customer_data.csv")

# Handle missing values
imputer = SimpleImputer(strategy="mean")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Scale numerical features
scaler = StandardScaler()
numerical_columns = ["age", "income", "tenure"]
data_imputed[numerical_columns] = scaler.fit_transform(data_imputed[numerical_columns])

print(data_imputed.head())
```

Slide 4: Exploratory Data Analysis (EDA)

EDA helps understand data characteristics, identify patterns, and uncover insights that guide feature engineering and model selection.

```python
import seaborn as sns

# Visualize distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(data=data_imputed, x="churn", kde=True)
plt.title("Distribution of Customer Churn")
plt.show()

# Correlation heatmap
correlation_matrix = data_imputed.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

Slide 5: Feature Engineering

Feature engineering involves creating new features or transforming existing ones to improve model performance and capture domain knowledge.

```python
import numpy as np

def create_interaction_features(df, feature1, feature2):
    return df[feature1] * df[feature2]

def bin_continuous_variable(df, column, bins):
    return pd.cut(df[column], bins=bins, labels=False)

# Create interaction feature
data_imputed["age_tenure_interaction"] = create_interaction_features(data_imputed, "age", "tenure")

# Bin continuous variable
data_imputed["income_bracket"] = bin_continuous_variable(data_imputed, "income", bins=5)

print(data_imputed[["age", "tenure", "age_tenure_interaction", "income", "income_bracket"]].head())
```

Slide 6: Model Selection

Choosing the right model depends on the problem type, data characteristics, and project requirements. It's essential to consider interpretability, performance, and computational resources.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Prepare data
X = data_imputed.drop("churn", axis=1)
y = data_imputed["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} - Accuracy: {score:.4f}")
```

Slide 7: Model Training and Hyperparameter Tuning

Training involves fitting the model to the data, while hyperparameter tuning optimizes model performance by adjusting its configuration.

```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10]
}

# Perform grid search
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

Slide 8: Model Evaluation

Evaluation assesses model performance using various metrics and techniques to ensure it meets project requirements and generalizes well to unseen data.

```python
from sklearn.metrics import confusion_matrix, classification_report

# Make predictions
y_pred = grid_search.best_estimator_.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))
```

Slide 9: Model Interpretation

Interpreting the model helps understand its decision-making process, which is crucial for building trust and gaining insights into the problem domain.

```python
import shap

# Create explainer
explainer = shap.TreeExplainer(grid_search.best_estimator_)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Plot summary
shap.summary_plot(shap_values[1], X_test, plot_type="bar")
plt.title("Feature Importance (SHAP Values)")
plt.show()
```

Slide 10: Model Deployment

Deployment involves integrating the trained model into production systems, ensuring scalability, reliability, and efficient inference.

```python
import joblib

# Save the model
joblib.dump(grid_search.best_estimator_, "churn_prediction_model.joblib")

# Function to load and use the model
def predict_churn(customer_data):
    model = joblib.load("churn_prediction_model.joblib")
    prediction = model.predict(customer_data)
    return "Churn" if prediction[0] == 1 else "Not Churn"

# Example usage
new_customer = X_test.iloc[0].values.reshape(1, -1)
result = predict_churn(new_customer)
print(f"Churn prediction: {result}")
```

Slide 11: Monitoring and Maintenance

Continuous monitoring and maintenance ensure the model's performance remains consistent over time and adapts to changing data distributions.

```python
import numpy as np
from scipy import stats

def detect_data_drift(reference_data, new_data, threshold=0.05):
    drift_detected = False
    for column in reference_data.columns:
        _, p_value = stats.ks_2samp(reference_data[column], new_data[column])
        if p_value < threshold:
            print(f"Drift detected in feature: {column}")
            drift_detected = True
    return drift_detected

# Simulate new data
new_data = X_test.()
new_data["age"] += np.random.normal(0, 5, size=len(new_data))

# Check for data drift
drift_detected = detect_data_drift(X_train, new_data)
if not drift_detected:
    print("No significant data drift detected")
```

Slide 12: Real-life Example: Customer Churn Prediction

This example demonstrates how machine learning can be applied to predict customer churn in a telecommunications company.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load telco customer churn data
telco_data = pd.read_csv("telco_customer_churn.csv")

# Preprocess data
telco_data["TotalCharges"] = pd.to_numeric(telco_data["TotalCharges"], errors="coerce")
telco_data.dropna(inplace=True)
telco_data = pd.get_dummies(telco_data, drop_first=True)

# Prepare features and target
X = telco_data.drop("Churn_Yes", axis=1)
y = telco_data["Churn_Yes"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

Slide 13: Real-life Example: Image Classification for Plant Disease Detection

This example shows how machine learning can be used to classify plant diseases from leaf images, aiding farmers in early detection and treatment.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'plant_disease_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'plant_disease_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build model
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(len(train_generator.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 14: Additional Resources

For further exploration of machine learning topics:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press)
2. "Pattern Recognition and Machine Learning" by Christopher Bishop (Springer)
3. "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron (O'Reilly Media)
4. ArXiv.org for latest research papers: [https://arxiv.org/list/cs.LG/recent](https://arxiv.org/list/cs.LG/recent) (Machine Learning category)
5. Coursera's Machine Learning Specialization by Andrew Ng
6. Fast.ai's Practical Deep Learning for Coders course

Remember to stay updated with the latest developments in the field and practice implementing algorithms on real-world datasets.
