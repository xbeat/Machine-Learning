## Git Essentials Your Cheat Sheet for Mastering Version Control
Slide 1: Git Essentials Introduction

Git is a distributed version control system that helps developers manage and track changes in their codebase. It allows for efficient collaboration, branching, and merging of code. This slideshow will cover essential Git commands and concepts, with practical Python examples to illustrate their usage.

```python
# Simple Python script to demonstrate Git usage
import os

def create_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)

def main():
    # Create a new file
    create_file('example.py', 'print("Hello, Git!")')
    
    # Simulate Git commands
    os.system('git init')
    os.system('git add example.py')
    os.system('git commit -m "Initial commit"')

if __name__ == '__main__':
    main()
```

Slide 2: Initializing a Git Repository

The `git init` command initializes a new Git repository in the current directory. It creates a hidden `.git` folder that stores all the version control information.

```python
import os
import subprocess

def init_git_repo(directory):
    os.chdir(directory)
    result = subprocess.run(['git', 'init'], capture_output=True, text=True)
    return result.stdout.strip()

# Example usage
print(init_git_repo('/path/to/your/project'))
```

Slide 3: Cloning a Repository

The `git clone` command creates a copy of an existing repository on your local machine. It downloads all files and history from the remote repository.

```python
import subprocess

def clone_repo(repo_url, destination):
    result = subprocess.run(['git', 'clone', repo_url, destination], capture_output=True, text=True)
    return result.stdout.strip()

# Example usage
print(clone_repo('https://github.com/example/repo.git', '/path/to/destination'))
```

Slide 4: Checking Repository Status

The `git status` command shows the current state of your working directory, including modified files, staged changes, and untracked files.

```python
import subprocess

def check_git_status():
    result = subprocess.run(['git', 'status'], capture_output=True, text=True)
    return result.stdout

# Example usage
print(check_git_status())
```

Slide 5: Staging Changes

The `git add` command stages changes for commit. You can stage specific files or all changes in the working directory.

```python
import subprocess

def stage_changes(files):
    if isinstance(files, str):
        files = [files]
    result = subprocess.run(['git', 'add'] + files, capture_output=True, text=True)
    return result.stdout

# Example usage
print(stage_changes(['file1.py', 'file2.py']))
print(stage_changes('.'))  # Stage all changes
```

Slide 6: Committing Changes

The `git commit` command records the staged changes in the repository history with a descriptive message.

```python
import subprocess

def commit_changes(message):
    result = subprocess.run(['git', 'commit', '-m', message], capture_output=True, text=True)
    return result.stdout

# Example usage
print(commit_changes("Add new feature: user authentication"))
```

Slide 7: Creating and Switching Branches

Branches allow you to work on different features or versions of your code simultaneously. The `git branch` command creates a new branch, and `git checkout` switches between branches.

```python
import subprocess

def create_branch(branch_name):
    result = subprocess.run(['git', 'branch', branch_name], capture_output=True, text=True)
    return result.stdout

def switch_branch(branch_name):
    result = subprocess.run(['git', 'checkout', branch_name], capture_output=True, text=True)
    return result.stdout

# Example usage
print(create_branch("feature-login"))
print(switch_branch("feature-login"))
```

Slide 8: Merging Branches

The `git merge` command combines changes from one branch into another, typically used to integrate feature branches back into the main branch.

```python
import subprocess

def merge_branch(branch_name):
    result = subprocess.run(['git', 'merge', branch_name], capture_output=True, text=True)
    return result.stdout

# Example usage
print(merge_branch("feature-login"))
```

Slide 9: Fetching and Pulling Remote Changes

The `git fetch` command downloads changes from a remote repository without merging them, while `git pull` fetches and merges changes in one step.

```python
import subprocess

def fetch_changes(remote='origin'):
    result = subprocess.run(['git', 'fetch', remote], capture_output=True, text=True)
    return result.stdout

def pull_changes(remote='origin', branch='main'):
    result = subprocess.run(['git', 'pull', remote, branch], capture_output=True, text=True)
    return result.stdout

# Example usage
print(fetch_changes())
print(pull_changes())
```

Slide 10: Pushing Changes to Remote

The `git push` command uploads your local commits to a remote repository, making your changes available to other collaborators.

```python
import subprocess

def push_changes(remote='origin', branch='main'):
    result = subprocess.run(['git', 'push', remote, branch], capture_output=True, text=True)
    return result.stdout

# Example usage
print(push_changes())
```

Slide 11: Viewing Commit History

The `git log` command displays the commit history of your repository, showing commit messages, authors, and timestamps.

```python
import subprocess

def view_commit_history(num_commits=5):
    result = subprocess.run(['git', 'log', f'-n {num_commits}', '--oneline'], capture_output=True, text=True)
    return result.stdout

# Example usage
print(view_commit_history())
```

Slide 12: Reverting Changes

The `git revert` command creates a new commit that undoes the changes made in a specific commit, allowing you to safely undo mistakes.

```python
import subprocess

def revert_commit(commit_hash):
    result = subprocess.run(['git', 'revert', commit_hash], capture_output=True, text=True)
    return result.stdout

# Example usage
print(revert_commit("abc123"))  # Replace with an actual commit hash
```

Slide 13: Tagging Releases

Git tags allow you to mark specific points in your repository's history, typically used for marking release versions.

```python
import subprocess

def create_tag(tag_name, message):
    result = subprocess.run(['git', 'tag', '-a', tag_name, '-m', message], capture_output=True, text=True)
    return result.stdout

# Example usage
print(create_tag("v1.0.0", "First stable release"))
```

Slide 14: Real-Life Example: Collaborative Project

In this example, we'll simulate a collaborative project where multiple developers work on different features.

```python
import subprocess
import os

def simulate_collaboration():
    # Developer 1: Create a new feature branch
    subprocess.run(['git', 'checkout', '-b', 'feature-user-profile'])
    with open('user_profile.py', 'w') as f:
        f.write('def get_user_profile():\n    return {"name": "John Doe", "email": "john@example.com"}')
    subprocess.run(['git', 'add', 'user_profile.py'])
    subprocess.run(['git', 'commit', '-m', "Add user profile feature"])
    
    # Developer 2: Work on another feature
    subprocess.run(['git', 'checkout', '-b', 'feature-data-analysis'])
    with open('data_analysis.py', 'w') as f:
        f.write('def analyze_data(data):\n    return sum(data) / len(data)')
    subprocess.run(['git', 'add', 'data_analysis.py'])
    subprocess.run(['git', 'commit', '-m', "Add data analysis feature"])
    
    # Merge features into main branch
    subprocess.run(['git', 'checkout', 'main'])
    subprocess.run(['git', 'merge', 'feature-user-profile'])
    subprocess.run(['git', 'merge', 'feature-data-analysis'])

# Run the simulation
simulate_collaboration()
print("Collaboration simulation completed.")
```

Slide 15: Additional Resources

For more in-depth information on Git and version control, consider exploring these resources:

1.  Official Git documentation: [https://git-scm.com/doc](https://git-scm.com/doc)
2.  Pro Git book (free online): [https://git-scm.com/book/en/v2](https://git-scm.com/book/en/v2)
3.  GitHub Learning Lab: [https://lab.github.com/](https://lab.github.com/)
4.  ArXiv paper on distributed version control systems: [https://arxiv.org/abs/1409.1882](https://arxiv.org/abs/1409.1882)

