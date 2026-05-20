## Versioning Strategies for Lightweight ML Projects
Slide 1: Version Control for Machine Learning Projects

Version control is essential for managing ML projects effectively. While Data Version Control (DVC) is a powerful tool, simpler approaches can sometimes be more appropriate. Let's explore various versioning strategies for ML projects, considering their pros and cons.

```python
import git
import os

def initialize_git_repo(path):
    if not os.path.exists(path):
        os.makedirs(path)
    repo = git.Repo.init(path)
    print(f"Initialized Git repository in {path}")
    return repo

# Usage
project_path = "./ml_project"
repo = initialize_git_repo(project_path)
```

Slide 2: Git-LFS for Large File Storage

Git Large File Storage (LFS) is a Git extension that helps manage large files by storing file contents on a remote server while keeping lightweight references in the Git repository.

```python
import subprocess

def setup_git_lfs():
    try:
        subprocess.run(["git", "lfs", "install"], check=True)
        print("Git LFS installed successfully")
    except subprocess.CalledProcessError:
        print("Error: Git LFS installation failed")

def track_large_file(file_pattern):
    try:
        subprocess.run(["git", "lfs", "track", file_pattern], check=True)
        print(f"Now tracking {file_pattern} with Git LFS")
    except subprocess.CalledProcessError:
        print(f"Error: Failed to track {file_pattern}")

# Usage
setup_git_lfs()
track_large_file("*.csv")
track_large_file("*.h5")
```

Slide 3: S3 Versioning for Dataset Management

Amazon S3 offers built-in versioning capabilities, which can be a simple yet effective way to manage dataset versions without additional tools.

```python
import boto3

def enable_s3_versioning(bucket_name):
    s3 = boto3.client('s3')
    try:
        s3.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        print(f"Versioning enabled for bucket: {bucket_name}")
    except Exception as e:
        print(f"Error enabling versioning: {str(e)}")

# Usage
enable_s3_versioning('my-ml-datasets')
```

Slide 4: DVC for Complex ML Workflows

Data Version Control (DVC) excels in managing complex ML pipelines, especially for larger teams working on iterative projects.

```python
import os
import subprocess

def initialize_dvc():
    try:
        subprocess.run(["dvc", "init"], check=True)
        print("DVC initialized successfully")
    except subprocess.CalledProcessError:
        print("Error: DVC initialization failed")

def add_data_to_dvc(data_path):
    try:
        subprocess.run(["dvc", "add", data_path], check=True)
        print(f"Added {data_path} to DVC")
    except subprocess.CalledProcessError:
        print(f"Error: Failed to add {data_path} to DVC")

# Usage
os.chdir("./ml_project")  # Assuming we're in the project directory
initialize_dvc()
add_data_to_dvc("data/large_dataset.csv")
```

Slide 5: Comparing Versioning Approaches

Let's compare Git-LFS, S3 Versioning, and DVC to understand their strengths and use cases.

```python
import pandas as pd

comparison_data = {
    'Feature': ['Large File Handling', 'Integration with Git', 'Remote Storage', 'Pipeline Management'],
    'Git-LFS': ['Good', 'Excellent', 'Limited', 'No'],
    'S3 Versioning': ['Excellent', 'Poor', 'Excellent', 'No'],
    'DVC': ['Excellent', 'Good', 'Excellent', 'Excellent']
}

df = pd.DataFrame(comparison_data)
print(df.to_string(index=False))
```

Output:

```
       Feature Git-LFS S3 Versioning    DVC
Large File Handling    Good     Excellent Excellent
Integration with Git Excellent         Poor     Good
    Remote Storage  Limited     Excellent Excellent
Pipeline Management      No            No Excellent
```

Slide 6: When to Use Git-LFS

Git-LFS is ideal for projects with occasional large files that need to be version-controlled alongside code.

```python
def should_use_git_lfs(file_sizes, team_size, git_integration_importance):
    large_files = sum(1 for size in file_sizes if size > 100 * 1024 * 1024)  # Files larger than 100MB
    
    if large_files > 0 and team_size < 5 and git_integration_importance > 7:
        return True
    return False

# Example usage
project_files = [50 * 1024 * 1024, 150 * 1024 * 1024, 200 * 1024 * 1024]  # File sizes in bytes
team_members = 3
git_importance = 9  # On a scale of 1-10

use_git_lfs = should_use_git_lfs(project_files, team_members, git_importance)
print(f"Should use Git-LFS: {use_git_lfs}")
```

Output:

```
Should use Git-LFS: True
```

Slide 7: When to Use S3 Versioning

S3 Versioning is suitable for projects that primarily need dataset versioning without complex pipelines.

```python
def should_use_s3_versioning(data_size_gb, update_frequency, need_for_pipelines):
    if data_size_gb > 100 and update_frequency < 7 and not need_for_pipelines:
        return True
    return False

# Example usage
dataset_size = 500  # GB
updates_per_week = 2
require_pipelines = False

use_s3 = should_use_s3_versioning(dataset_size, updates_per_week, require_pipelines)
print(f"Should use S3 Versioning: {use_s3}")
```

Output:

```
Should use S3 Versioning: True
```

Slide 8: When to Use DVC

DVC shines in projects with complex ML workflows, frequent dataset changes, and larger teams.

```python
def should_use_dvc(team_size, pipeline_complexity, data_change_frequency):
    score = (team_size * 0.3) + (pipeline_complexity * 0.4) + (data_change_frequency * 0.3)
    return score > 7

# Example usage
team_members = 10
pipeline_score = 8  # On a scale of 1-10
data_updates = 9  # On a scale of 1-10

use_dvc = should_use_dvc(team_members, pipeline_score, data_updates)
print(f"Should use DVC: {use_dvc}")
```

Output:

```
Should use DVC: True
```

Slide 9: Real-Life Example: Image Classification Project

Consider an image classification project with a moderate-sized dataset and infrequent updates.

```python
import os
from git import Repo

def setup_image_classification_project():
    # Initialize Git repository
    repo = Repo.init("./image_classifier")
    
    # Set up Git LFS for image files
    os.system("git lfs install")
    os.system("git lfs track '*.jpg' '*.png'")
    
    # Create directories
    os.makedirs("./image_classifier/data", exist_ok=True)
    os.makedirs("./image_classifier/models", exist_ok=True)
    
    # Create a sample script
    with open("./image_classifier/train.py", "w") as f:
        f.write("# Image classification training script\n")
    
    # Commit changes
    repo.index.add(["*"])
    repo.index.commit("Initial project setup with Git LFS")
    
    print("Image classification project set up with Git and Git LFS")

setup_image_classification_project()
```

Slide 10: Real-Life Example: NLP Model with Frequent Dataset Updates

For a Natural Language Processing project with frequent dataset changes and complex pipelines, DVC would be more appropriate.

```python
import os
import subprocess

def setup_nlp_project_with_dvc():
    # Initialize Git and DVC
    os.system("git init")
    os.system("dvc init")
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create a sample dataset and add to DVC
    with open("data/text_dataset.txt", "w") as f:
        f.write("Sample text data for NLP")
    os.system("dvc add data/text_dataset.txt")
    
    # Create a DVC pipeline
    with open("dvc.yaml", "w") as f:
        f.write("""
stages:
  preprocess:
    cmd: python preprocess.py
    deps:
      - data/text_dataset.txt
    outs:
      - data/processed_data.pkl
  train:
    cmd: python train.py
    deps:
      - data/processed_data.pkl
    outs:
      - models/nlp_model.pkl
        """)
    
    # Commit changes
    os.system("git add .")
    os.system('git commit -m "Set up NLP project with DVC"')
    
    print("NLP project set up with Git and DVC")

setup_nlp_project_with_dvc()
```

Slide 11: Hybrid Approach: Combining Versioning Strategies

In some cases, a hybrid approach combining multiple versioning strategies can provide the best of all worlds.

```python
import os
import subprocess

def setup_hybrid_ml_project():
    # Initialize Git and DVC
    os.system("git init")
    os.system("dvc init")
    
    # Set up Git LFS for large binary files
    os.system("git lfs install")
    os.system("git lfs track '*.h5' '*.pkl'")
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Use DVC for dataset versioning
    with open("data/dataset.csv", "w") as f:
        f.write("sample,label\n1,0\n2,1\n")
    os.system("dvc add data/dataset.csv")
    
    # Use Git LFS for model versioning
    with open("models/model.h5", "wb") as f:
        f.write(b"dummy model data")
    
    # Create a DVC pipeline
    with open("dvc.yaml", "w") as f:
        f.write("""
stages:
  train:
    cmd: python train.py
    deps:
      - data/dataset.csv
    outs:
      - models/model.h5
        """)
    
    # Commit changes
    os.system("git add .")
    os.system('git commit -m "Set up hybrid ML project"')
    
    print("Hybrid ML project set up with Git, Git LFS, and DVC")

setup_hybrid_ml_project()
```

Slide 12: Best Practices for ML Project Versioning

Regardless of the chosen versioning strategy, following best practices ensures efficient project management.

```python
def version_control_best_practices():
    practices = {
        "Use meaningful commit messages": lambda: "git commit -m 'Add feature X to improve Y'",
        "Create branches for experiments": lambda: "git checkout -b experiment/new_model",
        "Tag important milestones": lambda: "git tag -a v1.0 -m 'First stable release'",
        "Document data lineage": lambda: "# In README.md: Data sourced from X, processed on Y date",
        "Automate versioning where possible": lambda: "# In CI script: git describe --tags --always > VERSION"
    }
    
    for practice, example in practices.items():
        print(f"{practice}:")
        print(f"Example: {example()}\n")

version_control_best_practices()
```

Slide 13: Conclusion: Choosing the Right Versioning Strategy

The choice of versioning strategy depends on project complexity, team size, and specific requirements. While DVC offers comprehensive solutions for complex ML workflows, simpler approaches like Git-LFS or S3 versioning can be equally effective for smaller projects or those with less frequent data changes.

```python
def recommend_versioning_strategy(project_size, data_change_frequency, pipeline_complexity):
    if project_size == "small" and data_change_frequency == "low":
        return "Git + Git-LFS"
    elif project_size == "medium" and pipeline_complexity == "low":
        return "Git + S3 Versioning"
    elif project_size == "large" or pipeline_complexity == "high":
        return "DVC"
    else:
        return "Consider a hybrid approach"

# Example usage
project_scenarios = [
    ("small", "low", "low"),
    ("medium", "medium", "low"),
    ("large", "high", "high")
]

for size, freq, complexity in project_scenarios:
    recommendation = recommend_versioning_strategy(size, freq, complexity)
    print(f"Project: size={size}, data changes={freq}, complexity={complexity}")
    print(f"Recommended strategy: {recommendation}\n")
```

Slide 14: Additional Resources

For more information on ML project versioning:

1.  "Data Version Control with DVC" - ArXiv:2012.09951 ([https://arxiv.org/abs/2012.09951](https://arxiv.org/abs/2012.09951))
2.  "Versioning for End-to-End Machine Learning Projects" - ArXiv:2006.02371 ([https://arxiv.org/abs/2006.02371](https://arxiv.org/abs/2006.02371))

These papers provide in-depth discussions on versioning strategies for ML projects, including comparisons of different tools and methodologies.

