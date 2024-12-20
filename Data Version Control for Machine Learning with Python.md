## Data Version Control for Machine Learning with Python
Slide 1: Introduction to Data Version Control (DVC) in Machine Learning

Data Version Control (DVC) is a powerful tool for managing and versioning datasets and machine learning models. It helps data scientists and ML engineers track changes, collaborate efficiently, and reproduce experiments. DVC integrates seamlessly with Git, extending version control capabilities to large files and directories.

```python
import dvc.api

# Access a specific version of a dataset
with dvc.api.open('data/dataset.csv', rev='v1.0') as f:
    data = f.read()
    print(f"Dataset version 1.0 size: {len(data)} bytes")
```

Slide 2: Setting Up DVC

To get started with DVC, you need to initialize it in your project directory. This creates a .dvc folder to store configuration and cache files. DVC works alongside Git, so make sure your project is already a Git repository.

```python
import os

# Initialize DVC in your project
os.system('dvc init')

# Add files to DVC
os.system('dvc add data/large_dataset.csv')

# Commit changes to Git
os.system('git add data/.gitignore data/large_dataset.csv.dvc')
os.system('git commit -m "Add large dataset to DVC"')

print("DVC initialized and dataset added")
```

Slide 3: Tracking Data Changes

DVC allows you to track changes in your datasets over time. When you modify a tracked file, DVC detects the changes and updates the corresponding .dvc file. This enables you to version your data alongside your code.

```python
import pandas as pd
import os

# Load and modify the dataset
df = pd.read_csv('data/large_dataset.csv')
df['new_column'] = df['existing_column'] * 2
df.to_csv('data/large_dataset.csv', index=False)

# Update DVC tracking
os.system('dvc add data/large_dataset.csv')
os.system('git add data/large_dataset.csv.dvc')
os.system('git commit -m "Update dataset with new column"')

print("Dataset updated and changes tracked in DVC")
```

Slide 4: Managing Model Versions

DVC isn't just for data; it's also great for versioning machine learning models. By tracking model files, you can easily switch between different versions of your model during development and deployment.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Train a model
X, y = load_data()  # Assume this function loads your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/random_forest_v1.joblib')

# Add model to DVC
os.system('dvc add models/random_forest_v1.joblib')
os.system('git add models/.gitignore models/random_forest_v1.joblib.dvc')
os.system('git commit -m "Add Random Forest model v1"')

print("Model trained, saved, and added to DVC")
```

Slide 5: Reproducing Experiments

One of DVC's key features is its ability to reproduce experiments. By using DVC pipelines, you can define the steps of your ML workflow and easily rerun them with different data or parameters.

```python
# dvc.yaml
stages:
  prepare:
    cmd: python prepare.py
    deps:
      - data/raw_data.csv
    outs:
      - data/prepared_data.csv
  train:
    cmd: python train.py
    deps:
      - data/prepared_data.csv
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false

# In your terminal:
# $ dvc repro
```

Slide 6: Comparing Experiments

DVC provides tools to compare different runs of your experiments. This is crucial for understanding how changes in data or hyperparameters affect your model's performance.

```python
import os
import json

# Run two experiments with different parameters
os.system('python train.py --learning_rate 0.01 --model_name exp1')
os.system('python train.py --learning_rate 0.1 --model_name exp2')

# Compare experiments
os.system('dvc exp show')

# Load and print metrics
with open('metrics_exp1.json', 'r') as f:
    metrics_exp1 = json.load(f)
with open('metrics_exp2.json', 'r') as f:
    metrics_exp2 = json.load(f)

print(f"Experiment 1 accuracy: {metrics_exp1['accuracy']}")
print(f"Experiment 2 accuracy: {metrics_exp2['accuracy']}")
```

Slide 7: Remote Storage Integration

DVC supports various remote storage options, allowing you to store and share large datasets and models efficiently. This is particularly useful for collaborating on ML projects.

```python
import os

# Add a remote storage (e.g., S3 bucket)
os.system('dvc remote add -d myremote s3://mybucket/dvcstore')

# Push data to remote storage
os.system('dvc push')

# Pull data from remote storage
os.system('dvc pull')

print("Data synchronized with remote storage")
```

Slide 8: Visualizing the ML Pipeline

DVC allows you to visualize your ML pipeline, making it easier to understand the workflow and dependencies between different stages of your project.

```python
import os
import networkx as nx
import matplotlib.pyplot as plt

# Generate DAG representation of the pipeline
os.system('dvc dag --dot > pipeline.dot')

# Read the DOT file and create a graph
G = nx.drawing.nx_pydot.read_dot('pipeline.dot')

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=10, arrows=True)
plt.title("ML Pipeline Visualization")
plt.axis('off')
plt.show()
```

Slide 9: Real-Life Example: Image Classification Pipeline

Let's consider a real-life example of using DVC in an image classification project. We'll create a pipeline for preprocessing images, training a model, and evaluating its performance.

```python
# dvc.yaml
stages:
  preprocess:
    cmd: python preprocess_images.py data/raw_images data/processed_images
    deps:
      - data/raw_images
    outs:
      - data/processed_images
  train:
    cmd: python train_model.py data/processed_images models/image_classifier.h5
    deps:
      - data/processed_images
    outs:
      - models/image_classifier.h5
  evaluate:
    cmd: python evaluate_model.py models/image_classifier.h5 data/test_images metrics.json
    deps:
      - models/image_classifier.h5
      - data/test_images
    metrics:
      - metrics.json:
          cache: false

# In your terminal:
# $ dvc repro
```

Slide 10: Real-Life Example: Natural Language Processing Project

Another practical example of using DVC is in a Natural Language Processing (NLP) project. Let's create a pipeline for text preprocessing, training a sentiment analysis model, and evaluating its performance.

```python
# dvc.yaml
stages:
  preprocess:
    cmd: python preprocess_text.py data/raw_texts.txt data/processed_texts.txt
    deps:
      - data/raw_texts.txt
    outs:
      - data/processed_texts.txt
  train:
    cmd: python train_sentiment_model.py data/processed_texts.txt models/sentiment_model.pkl
    deps:
      - data/processed_texts.txt
    outs:
      - models/sentiment_model.pkl
  evaluate:
    cmd: python evaluate_sentiment.py models/sentiment_model.pkl data/test_texts.txt metrics.json
    deps:
      - models/sentiment_model.pkl
      - data/test_texts.txt
    metrics:
      - metrics.json:
          cache: false

# In your terminal:
# $ dvc repro
```

Slide 11: Handling Large Datasets

DVC excels at managing large datasets that are impractical to store directly in Git. It uses file hashing to track changes efficiently and allows for partial dataset updates.

```python
import os
import hashlib

def hash_file(filename):
    h = hashlib.sha256()
    with open(filename, 'rb') as file:
        chunk = 0
        while chunk != b'':
            chunk = file.read(1024)
            h.update(chunk)
    return h.hexdigest()

# Add a large dataset to DVC
os.system('dvc add data/large_dataset.csv')

# Get the hash of the dataset
dataset_hash = hash_file('data/large_dataset.csv')
print(f"Dataset hash: {dataset_hash}")

# Update only a portion of the dataset
with open('data/large_dataset.csv', 'a') as f:
    f.write("new,data,row\n")

# Re-add to DVC to track changes
os.system('dvc add data/large_dataset.csv')

new_hash = hash_file('data/large_dataset.csv')
print(f"New dataset hash: {new_hash}")
```

Slide 12: Collaborative Workflow with DVC

DVC enhances collaboration in ML projects by allowing team members to share data and model versions easily. Here's an example of a collaborative workflow using DVC and Git.

```python
import os

# Developer A: Create and push a new experiment
os.system('git checkout -b new_feature')
os.system('python train_model.py --new_param value')
os.system('dvc add model.pkl')
os.system('git add model.pkl.dvc')
os.system('git commit -m "Train model with new parameter"')
os.system('git push origin new_feature')
os.system('dvc push')

# Developer B: Pull and reproduce the experiment
os.system('git pull origin new_feature')
os.system('git checkout new_feature')
os.system('dvc pull')
os.system('dvc repro')

print("Collaboration workflow completed")
```

Slide 13: Metrics Tracking and Comparison

DVC allows you to track and compare metrics across different experiments, making it easier to identify the best-performing models.

```python
import os
import json
import matplotlib.pyplot as plt

# Run multiple experiments
experiments = ['exp1', 'exp2', 'exp3']
for exp in experiments:
    os.system(f'python train.py --experiment {exp}')
    os.system(f'dvc exp save {exp}')

# Compare metrics
os.system('dvc exp show --csv > metrics.csv')

# Visualize metrics
metrics = {}
with open('metrics.csv', 'r') as f:
    for line in f:
        exp, accuracy = line.strip().split(',')
        metrics[exp] = float(accuracy)

plt.bar(metrics.keys(), metrics.values())
plt.title('Model Accuracy Comparison')
plt.xlabel('Experiment')
plt.ylabel('Accuracy')
plt.show()
```

Slide 14: Continuous Integration with DVC

Integrating DVC into your Continuous Integration (CI) pipeline ensures that your data and model versions are consistently tracked and validated alongside your code.

```python
# .github/workflows/dvc_ci.yml

name: DVC CI

on: [push]

jobs:
  dvc-job:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install dvc
          pip install -r requirements.txt
      - name: Run DVC repro
        run: dvc repro
      - name: Check metrics
        run: python check_metrics.py

# In your project:
# check_metrics.py

import json

with open('metrics.json', 'r') as f:
    metrics = json.load(f)

assert metrics['accuracy'] > 0.8, "Model accuracy is below threshold"
```

Slide 15: Additional Resources

For those interested in diving deeper into Data Version Control for Machine Learning, here are some valuable resources:

1. DVC Documentation: [https://dvc.org/doc](https://dvc.org/doc)
2. "Versioning Data Science: Strategies for Version Control in Data-Intensive Projects" (ArXiv:2103.05822): [https://arxiv.org/abs/2103.05822](https://arxiv.org/abs/2103.05822)
3. "A Survey on Reproducibility by Evaluating Deep Reinforcement Learning Algorithms on Real-World Robots" (ArXiv:1909.03772): [https://arxiv.org/abs/1909.03772](https://arxiv.org/abs/1909.03772)
4. DVC GitHub Repository: [https://github.com/iterative/dvc](https://github.com/iterative/dvc)

These resources provide in-depth information on DVC usage, best practices, and its application in real-world machine learning projects.

