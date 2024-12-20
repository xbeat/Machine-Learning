## Experiment Tracking in Machine Learning with Python
Slide 1: Introduction to Experiment Tracking in Machine Learning

Experiment tracking is a crucial aspect of machine learning projects, enabling data scientists to organize, compare, and reproduce their work efficiently. It involves systematically recording parameters, metrics, and artifacts for each experiment run. This practice is essential for maintaining a clear history of your ML development process and facilitating collaboration among team members.

```python
import mlflow

# Start an MLflow experiment
mlflow.set_experiment("my_first_experiment")

with mlflow.start_run():
    # Log a parameter
    mlflow.log_param("learning_rate", 0.01)
    
    # Log a metric
    mlflow.log_metric("accuracy", 0.85)
    
    # Log an artifact (e.g., a model file)
    mlflow.log_artifact("model.pkl")
```

Slide 2: Setting Up MLflow for Experiment Tracking

MLflow is a popular open-source platform for managing the ML lifecycle, including experiment tracking. To get started, install MLflow using pip and import it into your Python script. MLflow provides a simple API for logging parameters, metrics, and artifacts during your experiment runs.

```python
# Install MLflow
!pip install mlflow

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the tracking URI (local file store in this case)
mlflow.set_tracking_uri("file:./mlruns")

# Set the experiment name
mlflow.set_experiment("random_forest_classifier")

# Load your data (assuming X and y are already defined)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow run
with mlflow.start_run():
    # Define and train the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model
    mlflow.sklearn.log_model(rf, "random_forest_model")

print(f"Model accuracy: {accuracy}")
```

Slide 3: Logging Parameters in MLflow

Parameters are input variables that define the configuration of your machine learning model or experiment. Logging parameters allows you to keep track of different configurations and their impact on model performance. MLflow provides a simple method to log parameters during an experiment run.

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier

# Start an MLflow run
with mlflow.start_run():
    # Define model parameters
    n_estimators = 100
    max_depth = 5
    min_samples_split = 2
    
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    
    # Create and train the model
    rf = RandomForestClassifier(n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                min_samples_split=min_samples_split)
    rf.fit(X_train, y_train)

    # ... (continue with model evaluation and logging metrics)
```

Slide 4: Logging Metrics in MLflow

Metrics are quantitative measures used to evaluate the performance of your machine learning model. Common metrics include accuracy, precision, recall, and F1-score. MLflow allows you to log these metrics during your experiment runs, making it easy to compare different models or configurations.

```python
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming you have already trained your model and made predictions

# Start an MLflow run
with mlflow.start_run():
    # Calculate various metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
```

Slide 5: Logging Artifacts in MLflow

Artifacts are files associated with your machine learning experiment, such as trained models, plots, or data files. MLflow provides functionality to log these artifacts, making it easy to store and retrieve them later. This is particularly useful for model versioning and reproducibility.

```python
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming you have already trained your model and made predictions

# Start an MLflow run
with mlflow.start_run():
    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    
    # Log the plot as an artifact
    mlflow.log_artifact('confusion_matrix.png')
    
    # Log the trained model as an artifact
    mlflow.sklearn.log_model(model, "trained_model")

print("Artifacts logged successfully")
```

Slide 6: Organizing Experiments with MLflow

MLflow allows you to organize your machine learning experiments into logical groups. This is particularly useful when working on multiple projects or comparing different approaches within the same project. You can create and set experiments using MLflow's API.

```python
import mlflow

# Set up a new experiment
mlflow.set_experiment("image_classification_project")

# Run multiple experiments within the same project
for model_type in ['cnn', 'resnet', 'vgg']:
    with mlflow.start_run(run_name=f"{model_type}_experiment"):
        # Your model training code here
        # ...
        
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("epochs", 10)
        
        # Log metrics
        mlflow.log_metric("accuracy", 0.85)
        
        # Log model
        mlflow.pytorch.log_model(model, "trained_model")

print("All experiments completed and logged")
```

Slide 7: Querying Experiment Results

MLflow provides a powerful API for querying and analyzing your experiment results. You can retrieve runs, compare metrics across different experiments, and even load saved models for further analysis or deployment.

```python
import mlflow
from mlflow.tracking import MlflowClient

# Create a client
client = MlflowClient()

# Get the experiment by name
experiment = client.get_experiment_by_name("image_classification_project")

# Get all runs for the experiment
runs = client.search_runs(experiment_ids=[experiment.experiment_id])

# Print results
for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"Model Type: {run.data.params['model_type']}")
    print(f"Accuracy: {run.data.metrics['accuracy']}")
    print("---")

# Load the best model (assuming the highest accuracy is best)
best_run = max(runs, key=lambda r: r.data.metrics['accuracy'])
best_model = mlflow.pytorch.load_model(f"runs:/{best_run.info.run_id}/trained_model")

print(f"Best model loaded from run: {best_run.info.run_id}")
```

Slide 8: Visualizing Experiment Results

MLflow's UI provides a convenient way to visualize and compare experiment results. However, you can also create custom visualizations using libraries like matplotlib or plotly to gain deeper insights into your experiments.

```python
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Fetch experiment runs
experiment = mlflow.get_experiment_by_name("image_classification_project")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Create a DataFrame from the runs
df = pd.DataFrame({
    'model_type': runs['params.model_type'],
    'accuracy': runs['metrics.accuracy']
})

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(df['model_type'], df['accuracy'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model Type')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Add value labels on top of each bar
for i, v in enumerate(df['accuracy']):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Log the plot as an artifact in a new run
with mlflow.start_run(run_name="result_visualization"):
    mlflow.log_artifact('model_comparison.png')

print("Visualization created and logged as an artifact")
```

Slide 9: Real-Life Example: Image Classification

Let's consider a real-life example of using MLflow for experiment tracking in an image classification project. We'll use a pre-trained ResNet model and fine-tune it on a custom dataset of animal images.

```python
import mlflow
import mlflow.pytorch
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Set up MLflow experiment
mlflow.set_experiment("animal_classification")

# Define transformations and load data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root='./animal_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))

# Training loop
with mlflow.start_run():
    mlflow.log_param("model", "ResNet50")
    mlflow.log_param("epochs", 5)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        
        mlflow.log_metric("loss", epoch_loss, step=epoch)
        mlflow.log_metric("accuracy", epoch_acc, step=epoch)
    
    mlflow.pytorch.log_model(model, "animal_classifier")

print("Training completed and model logged")
```

Slide 10: Real-Life Example: Natural Language Processing

In this example, we'll use MLflow to track experiments for a sentiment analysis task using different text classification models. We'll compare the performance of a simple Naive Bayes classifier and a more advanced BERT-based model.

```python
import mlflow
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Sample dataset (replace with your own data)
texts = ["I love this product!", "This is terrible.", "Neutral opinion."]
labels = [1, 0, 2]  # 1: positive, 0: negative, 2: neutral

# Set up MLflow experiment
mlflow.set_experiment("sentiment_analysis")

# Naive Bayes experiment
with mlflow.start_run(run_name="naive_bayes"):
    mlflow.log_param("model", "Naive Bayes")
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    
    nb_model = MultinomialNB()
    nb_model.fit(X, labels)
    
    nb_predictions = nb_model.predict(X)
    nb_accuracy = accuracy_score(labels, nb_predictions)
    nb_f1 = f1_score(labels, nb_predictions, average='weighted')
    
    mlflow.log_metric("accuracy", nb_accuracy)
    mlflow.log_metric("f1_score", nb_f1)
    mlflow.sklearn.log_model(nb_model, "naive_bayes_model")

# BERT experiment
with mlflow.start_run(run_name="bert"):
    mlflow.log_param("model", "BERT")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    
    sentiment_pipeline = pipeline("sentiment-analysis", model=bert_model, tokenizer=tokenizer)
    
    bert_predictions = [sentiment_pipeline(text)[0]['label'] for text in texts]
    bert_predictions = [int(label.split('_')[-1]) for label in bert_predictions]
    
    bert_accuracy = accuracy_score(labels, bert_predictions)
    bert_f1 = f1_score(labels, bert_predictions, average='weighted')
    
    mlflow.log_metric("accuracy", bert_accuracy)
    mlflow.log_metric("f1_score", bert_f1)
    mlflow.pytorch.log_model(bert_model, "bert_model")

print("Experiments completed and models logged")
```

Slide 11: Integrating MLflow with Other Tools

MLflow can be integrated with various other tools in the machine learning ecosystem to enhance your experiment tracking workflow. Here's an example of how to integrate MLflow with Optuna, a hyperparameter optimization framework.

```python
import mlflow
import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLflow experiment
mlflow.set_experiment("rf_hyperparameter_tuning")

def objective(trial):
    with mlflow.start_run(nested=True):
        # Define hyperparameters to optimize
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        
        # Create and train the model
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_params({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split
        })
        mlflow.log_metric('accuracy', accuracy)
        
        return accuracy

# Create and run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Log the best parameters and model
with mlflow.start_run():
    best_params = study.best_params
    best_accuracy = study.best_value
    
    mlflow.log_params(best_params)
    mlflow.log_metric('best_accuracy', best_accuracy)
    
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    mlflow.sklearn.log_model(best_model, 'best_model')

print(f"Best parameters: {best_params}")
print(f"Best accuracy: {best_accuracy}")
```

Slide 12: Version Control for ML Experiments

MLflow can be used in conjunction with version control systems like Git to maintain a comprehensive history of your machine learning experiments. This approach allows you to track both code changes and experiment results.

```python
import mlflow
import git
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Get Git repository information
repo = git.Repo(search_parent_directories=True)
commit_hash = repo.head.object.hexsha

# Load data
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLflow experiment
mlflow.set_experiment("boston_housing_prediction")

# Start MLflow run
with mlflow.start_run():
    # Log Git commit hash
    mlflow.log_param("git_commit", commit_hash)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("mse", mse)
    
    # Log model
    mlflow.sklearn.log_model(model, "linear_regression_model")

print(f"Experiment logged with commit: {commit_hash}")
print(f"Mean Squared Error: {mse}")
```

Slide 13: Reproducibility with MLflow

One of the key benefits of using MLflow is ensuring reproducibility of your machine learning experiments. MLflow allows you to log not only your model and results but also the entire environment, making it easier to recreate experiments later.

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Set up MLflow experiment
mlflow.set_experiment("reproducible_experiment")

# Start MLflow run
with mlflow.start_run():
    # Define and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Log conda environment
    mlflow.sklearn.log_model(model, "model", conda_env="conda.yaml")

print("Experiment logged with environment details")

# To reproduce the experiment later:
# loaded_model = mlflow.sklearn.load_model("runs:/<run_id>/random_forest_model")
```

Slide 14: Scaling MLflow for Team Collaboration

As your machine learning projects grow, you may need to scale MLflow to support team collaboration. This can be achieved by setting up a centralized MLflow tracking server that all team members can access.

```python
import mlflow
from mlflow.tracking import MlflowClient

# Set up connection to remote tracking server
mlflow.set_tracking_uri("http://your-mlflow-server:5000")

# Set experiment (creates a new one if it doesn't exist)
mlflow.set_experiment("team_collaboration_project")

# Start a run
with mlflow.start_run():
    # Your experiment code here
    # ...
    
    # Log parameters, metrics, and artifacts as usual
    mlflow.log_param("param_name", value)
    mlflow.log_metric("metric_name", value)
    mlflow.log_artifact("local/path/to/artifact")

# Query experiments and runs
client = MlflowClient()
experiments = client.list_experiments()
for exp in experiments:
    runs = client.search_runs(exp.experiment_id)
    for run in runs:
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"Status: {run.info.status}")
        print("Parameters:")
        for key, value in run.data.params.items():
            print(f"  {key}: {value}")
        print("Metrics:")
        for key, value in run.data.metrics.items():
            print(f"  {key}: {value}")
        print("---")
```

Slide 15: Additional Resources

For more information on experiment tracking in machine learning using Python and MLflow, consider exploring the following resources:

1. MLflow Documentation: [https://www.mlflow.org/docs/latest/index.html](https://www.mlflow.org/docs/latest/index.html)
2. "A Survey on Experiment Databases for Machine Learning" (ArXiv:2109.09028): [https://arxiv.org/abs/2109.09028](https://arxiv.org/abs/2109.09028)
3. "Towards Reproducible Machine Learning Research: A Review" (ArXiv:2103.03820): [https://arxiv.org/abs/2103.03820](https://arxiv.org/abs/2103.03820)
4. "MLOps: Continuous delivery and automation pipelines in machine learning" (ArXiv:2006.08528): [https://arxiv.org/abs/2006.08528](https://arxiv.org/abs/2006.08528)

These resources provide in-depth information on best practices, methodologies, and tools for effective experiment tracking and reproducibility in machine learning projects.

