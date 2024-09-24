## Leveraging Git From Narrative to Refactor
Slide 1: Interactive Rebase

Interactive rebase is a powerful Git feature that allows developers to rewrite commit history. This tool is essential for maintaining a clean and organized Git history, especially in collaborative environments.

Interactive rebase works by allowing you to modify, combine, or delete commits before they are applied to the target branch. This process can be visualized as follows:

* Original commits → Interactive rebase → Modified commits
* Messy history → Clean up → Clear narrative

Here's how to perform an interactive rebase:

```bash
git rebase -i HEAD~3
```

This command will open an editor where you can choose actions for the last three commits:

* pick → Keep the commit as is
* reword → Change the commit message
* squash → Combine with previous commit
* drop → Remove the commit

Interactive rebase is particularly useful for:

* Cleaning up work-in-progress commits → Creating a coherent feature history
* Fixing typos in commit messages → Improving project documentation
* Combining related commits → Simplifying code review process

Slide 2: Git Stashing

Git stashing is a feature that allows developers to temporarily save uncommitted changes and revert to a clean working directory. This is particularly useful when you need to switch contexts quickly without committing half-finished work.

The stashing process can be visualized as:

* Uncommitted changes → Git stash → Clean working directory
* Stashed changes → Git stash apply → Restored working state

Here are some common stash commands:

```bash
git stash save "Work in progress on feature X"
git stash list
git stash apply stash@{0}
git stash drop stash@{0}
git stash pop
```

Stashing is beneficial in scenarios such as:

* Urgent bug fix → Stash current work → Switch to bugfix branch
* Pull latest changes → Stash local modifications → Apply stash after pull
* Experiment with different approaches → Stash each attempt → Compare results

Remember that stashes are stored locally and are not pushed to remote repositories, making them a personal tool for managing your workflow.

Slide 3: Git Hooks

Git hooks are custom scripts that automatically run at certain points in Git's execution. They allow developers to automate tasks, enforce policies, and customize their Git workflow.

The flow of Git hooks can be represented as:

* Git event → Trigger hook → Execute custom script
* Commit attempt → Pre-commit hook → Code style check

Git provides various hook points, including:

* pre-commit → Run before a commit is created
* post-commit → Execute after a commit is created
* pre-push → Run before pushing commits to a remote
* post-merge → Execute after a successful merge

Here's an example of a simple pre-commit hook that checks for trailing whitespace:

```bash
#!/bin/sh
git diff --check --cached || exit 1
```

To use this hook, save it as `.git/hooks/pre-commit` and make it executable.

Git hooks enable workflows such as:

* Code style enforcement → Consistent codebase → Improved readability
* Automated testing → Pre-push hook → Prevent broken code from being pushed
* Ticket number validation → Commit-msg hook → Ensure proper commit messages

Hooks are powerful tools for maintaining code quality and streamlining development processes.

Slide 4: Cherry-Picking Commits

Cherry-picking in Git allows developers to apply specific commits from one branch to another. This feature is particularly useful when you want to selectively incorporate changes without merging entire branches.

The cherry-picking process can be visualized as:

* Source branch → Cherry-pick commit → Target branch
* Bugfix in feature branch → Cherry-pick to main → Immediate fix deployment

To cherry-pick a commit, use the following command:

```bash
git cherry-pick <commit-hash>
```

Cherry-picking is beneficial in scenarios such as:

* Hotfix in development → Cherry-pick to production → Quick issue resolution
* Experimental feature → Cherry-pick successful parts → Integrate into main project
* Backporting fixes → Cherry-pick newer fixes → Apply to older versions

When cherry-picking, keep in mind:

* Potential conflicts → Manual resolution may be needed
* Duplicate commits → Can occur if cherry-picked commit is later merged
* Context-dependent changes → May require additional modifications in the target branch

Cherry-picking is a powerful tool for managing complex branching strategies and selectively applying changes across your project's history.

Slide 5: Git Reflog

Git reflog is a powerful recovery tool that records all changes to branch tips in a local repository. It acts as a safety net, allowing developers to recover from mistakes or find lost commits.

The reflog process can be visualized as:

* Git actions → Recorded in reflog → Recoverable history
* Accidental branch deletion → Check reflog → Restore lost commits

To view the reflog, use:

```bash
git reflog
```

Reflog is particularly useful in scenarios such as:

* Incorrect reset → Find previous HEAD → Recover lost work
* Experimental rebasing → Reflog shows original state → Easy to revert changes
* Branch deletion → Reflog retains commit hashes → Recreate branch

Here's how to recover a lost commit using reflog:

```bash
git checkout -b recovery-branch <commit-hash>
```

Remember that:

* Reflog is local → Not pushed to remote repositories
* Entries expire → By default, kept for 90 days
* Regular garbage collection → May remove unreachable objects

Reflog serves as a valuable tool for maintaining data integrity and recovering from potentially catastrophic mistakes in Git operations.

Slide 6: Sparse Checkout

Sparse checkout in Git allows developers to check out only a subset of files from a repository. This feature is particularly useful for working with large repositories or when you only need specific parts of a project.

The sparse checkout process can be visualized as:

* Full repository → Sparse checkout configuration → Partial working directory
* Monorepo structure → Checkout specific module → Focused development environment

To set up a sparse checkout:

```bash
git clone --no-checkout <repository-url>
cd <repository-directory>
git sparse-checkout init
git sparse-checkout set <path1> <path2>
git checkout
```

Sparse checkout is beneficial in scenarios such as:

* Large monorepo → Checkout only relevant modules → Improved performance
* Limited disk space → Partial checkout → Work on specific areas
* Complex project → Focus on particular components → Simplified workflow

When using sparse checkout:

* Be aware of dependencies → Ensure all necessary files are included
* Updates to sparse-checkout configuration → May require re-checkout
* Collaboration considerations → Communicate partial checkouts to team members

Sparse checkout enables more efficient work with large-scale projects by allowing developers to focus on specific areas without the overhead of the entire repository.

Slide 7: Git Bisect

Git bisect is a powerful debugging tool that uses a binary search algorithm to find the commit that introduced a bug. This feature is particularly useful when dealing with regressions in large codebases.

The bisect process can be visualized as:

* Known good commit → Binary search → Known bad commit → Identify bug-introducing commit
* Start bisect → Mark commits as good/bad → Narrow down problematic change

To use git bisect:

```bash
git bisect start
git bisect bad  # Current commit is bad
git bisect good <known-good-commit>
# Git will checkout a commit halfway between good and bad
# Test the commit and mark it as good or bad
git bisect good  # or git bisect bad
# Repeat until the first bad commit is found
git bisect reset  # to end the bisect session
```

Bisect is especially useful for:

* Regression bugs → Quickly identify cause → Efficient debugging
* Performance issues → Pinpoint problematic changes → Optimize codebase
* Feature implementation → Trace feature addition → Understand implementation history

To automate the process, you can use:

```bash
git bisect run <test-script>
```

This runs a script on each commit, automatically marking it as good or bad based on the script's exit code.

Git bisect significantly reduces the time and effort required to track down issues in large projects with extensive commit histories.

Slide 8: Git Blame

Git blame is a diagnostic tool that shows the author and commit information for each line in a file. This feature is invaluable for understanding the evolution of code and tracking down the origins of specific changes.

The blame process can be visualized as:

* File content → Git blame → Annotated file with commit info
* Code investigation → Identify last modifier → Understand change context

To use git blame:

```bash
git blame <filename>
```

Git blame is particularly useful for:

* Bug investigation → Identify when bug was introduced → Contact relevant developer
* Code review → Understand change history → Provide context-aware feedback
* Documentation → Track content changes → Verify information accuracy

Git blame output includes:

* Commit hash → Unique identifier for the change
* Author name → Who made the change
* Date → When the change was made
* Line number → Position in the file
* Line content → The actual code or text

To focus on specific lines or ignore whitespace changes:

```bash
git blame -L 10,20 <filename>  # Only show lines 10-20
git blame -w <filename>  # Ignore whitespace changes
```

Git blame helps developers understand the context and history of code changes, facilitating more effective collaboration and debugging processes.

Slide 9: Git Submodules

Git submodules allow you to include one Git repository as a subdirectory of another Git repository. This feature is useful for incorporating external dependencies or breaking down large projects into manageable components.

The submodule relationship can be visualized as:

* Main repository → Contains submodule → Points to specific commit in submodule repo
* Project → Includes library as submodule → Manages dependency versions

To add a submodule:

```bash
git submodule add <repository-url> <path>
git commit -m "Add submodule"
```

Submodules are beneficial for:

* Dependency management → Pin external libraries to specific versions → Ensure consistency
* Monorepo alternatives → Split large projects → Maintain separate versioning
* Code reuse → Share common components → Centralize updates

When working with submodules:

* Cloning a project with submodules → Requires extra steps to initialize and update submodules
* Updating submodules → Main repo tracks submodule commit → Requires explicit update and commit

To clone a repository with submodules:

```bash
git clone --recurse-submodules <repository-url>
```

To update submodules:

```bash
git submodule update --remote
git commit -am "Update submodules"
```

Submodules provide a powerful way to manage complex project structures and dependencies, but require careful handling to avoid confusion and ensure all team members are working with the correct versions.

Slide 10: Reverting Commits

Git revert is a safe way to undo changes introduced by a commit by creating a new commit that undoes those changes. This approach is particularly useful for maintaining a clear history of actions taken in the repository.

The revert process can be visualized as:

* Problematic commit → Git revert → New commit undoing changes
* Feature implementation → Discover issues → Revert to stable state

To revert a commit:

```bash
git revert <commit-hash>
```

Reverting is beneficial in scenarios such as:

* Production hotfix → Revert problematic change → Quick resolution without losing history
* Feature rollback → Revert merge commit → Remove feature while preserving work done
* Collaborative workflows → Safely undo changes → Maintain clear project history

When reverting:

* Merge commits → May require specifying a parent with -m option
* Multiple commits → Can be reverted in reverse order
* Conflicts → May occur and require manual resolution

To revert multiple commits:

```bash
git revert --no-commit <oldest-commit-hash>^..<newest-commit-hash>
git commit -m "Revert multiple commits"
```

Git revert provides a safe and transparent way to undo changes, making it an essential tool for managing project history and recovering from errors without disturbing the existing commit timeline.

Slide 11: Git Diff

Git diff is a powerful command that shows the differences between various Git objects, such as commits, branches, files, and more. This tool is essential for code review, understanding changes, and resolving conflicts.

The diff process can be visualized as:

* Object A → Git diff → Object B → Highlighted differences
* Working directory → Git diff → Staged changes → Review before commit

Basic usage of git diff:

```bash
git diff  # Show unstaged changes
git diff --staged  # Show staged changes
git diff <commit1> <commit2>  # Compare two commits
git diff <branch1>..<branch2>  # Compare two branches
```

Git diff is particularly useful for:

* Code review → Examine changes before committing → Ensure code quality
* Conflict resolution → Understand differences → Make informed merge decisions
* Feature comparison → Diff branches → Evaluate implementation approaches

Output of git diff includes:

* File names → Indicate which files have changed
* Hunks → Sections of the file that differ
* Line-by-line changes → Added lines ("+"), removed lines ("-"), and context

To customize diff output:

```bash
git diff --color-words  # Highlight word-level changes
git diff --stat  # Show a summary of changes
```

Understanding and effectively using git diff is crucial for maintaining code quality, facilitating collaboration, and making informed decisions about code changes throughout the development process.

Slide 12: Git Worktrees

Git worktrees allow you to check out multiple branches of the same repository into separate directories. This feature is particularly useful for working on different branches simultaneously without switching or stashing changes.

The worktree concept can be visualized as:

* Main repository → Add worktree → Separate directory with different branch
* Feature development → Create worktree for main → Easy comparison and testing

To create a new worktree:

```bash
git worktree add ../path-to-new-dir branch-name
```

Worktrees are beneficial for:

* Parallel development → Work on multiple branches → Increased productivity
* CI/CD pipelines → Separate worktrees for different stages → Isolated environments
* Code review → Check out PR in separate worktree → Easy testing and comparison

When using worktrees:

* Main repository → Remains unchanged → Worktrees are separate
* Git operations → Performed in individual worktrees → Changes reflected in main repo
* Deleting worktrees → Use `git worktree remove` → Cleans up references

To list current worktrees:

```bash
git worktree list
```

Git worktrees provide a flexible way to manage multiple working copies of a repository, enabling efficient parallel development and testing without the need for multiple clones or constant branch switching.

Slide 13: Squash Merges

Squash merging is a Git technique that combines all commits from a feature branch into a single commit when merging into the main branch. This approach helps maintain a clean and readable Git history.

The squash merge process can be visualized as:

* Feature branch (multiple commits) → Squash merge → Main branch (single commit)
* Detailed development history → Condensed for main branch → Clean project timeline

To perform a squash merge:

```bash
git checkout main
git merge --squash feature-branch
git commit -m "Implement feature X"
```

Squash merging is beneficial for:

* Clean history → Simplify main branch timeline → Easier to understand project evolution
* Code review → Focus on overall changes → Simplified review process
* Release management → Group related changes → Clear feature boundaries in history

When using squash merges:

* Original commits → Lost in main branch → Preserved in feature branch
* Rebasing → May be necessary before squashing → Ensure up-to-date with main
* Team communication → Agree on squash policy → Maintain consistent practices

To view the condensed changes before committing:

```bash
git diff --cached
```

Squash merging offers a way to maintain a clean and organized Git history while still preserving detailed development information in feature branches, striking a balance between comprehensive tracking and readability.

Slide 14: Git Aliases

Git aliases are custom shortcuts for Git commands, allowing developers to create their own commands or simplify complex operations. This feature enhances productivity by reducing typing and standardizing common workflows.

The alias creation process can be visualized as:

* Frequently used command → Create alias → Simplified workflow
* Complex Git operation → Custom alias → One-line execution

To create a Git alias:

```bash
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
```

Aliases are particularly useful for:

* Common operations → Reduce typing → Increase efficiency
* Complex workflows → Encapsulate in alias → Standardize team practices
* Custom commands → Combine multiple Git operations → Streamline processes

Example of a more complex alias:

```bash
git config --global alias.undo 'reset --soft HEAD~1'
```

This creates an 'undo' command that resets the last commit while keeping changes staged.

When using aliases:

* Shared configurations → Document aliases → Ensure team-wide understanding
* Shell commands → Prefix with '!' → Execute non-Git commands
* Alias management → Review and update regularly → Optimize for current workflows

Git aliases provide a powerful way to customize and optimize your Git experience, allowing for more efficient and consistent use of Git across individual and team workflows.

Slide 15: Further Exploration

While we've covered many advanced Git techniques, there are still more topics worth exploring to further enhance your Git mastery:

* Git Flow → Branching model for project management
* Git LFS → Managing large files in Git repositories
* Git Internals → Understanding Git's object model and operations
* Rebasing vs. Merging → Choosing the right integration strategy
* Git Patch → Creating and applying patches for code sharing
* Git Attributes → Customizing Git's behavior for specific files or directories
* Git Rerere → Reusing recorded conflict resolutions
* Git Refspecs → Advanced remote branch and tag management
* Git Bundle → Transferring Git data without a network
* Git Notes → Adding metadata to commits without changing history

These topics represent advanced Git concepts and techniques that can significantly improve your workflow and understanding of version control. Each of these areas offers unique benefits and use cases:

* Basic concept → Advanced application → Improved Git workflow
* Standard practices → Specialized tools → Enhanced productivity

As you continue to work with Git, exploring these topics will provide you with a more comprehensive toolkit for managing your projects efficiently and effectively.

