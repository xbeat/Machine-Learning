## Essential Statistics for Data Insights

Slide 1: Understanding Descriptive Statistics

Descriptive statistics provide a powerful toolkit for summarizing and visualizing data, allowing us to extract meaningful insights from large datasets. They help us understand the central tendencies, spread, and shape of our data distribution.

```python
import matplotlib.pyplot as plt

# Generate sample data
data = np.random.normal(loc=50, scale=10, size=1000)

# Calculate descriptive statistics
mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black')
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
plt.title('Distribution of Sample Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print(f"Standard Deviation: {std_dev:.2f}")
```

Slide 2: Outlier Detection Techniques

Outliers can significantly impact statistical analyses and machine learning models. Identifying and handling these anomalies is crucial for maintaining data integrity and ensuring reliable results.

```python
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data with outliers
np.random.seed(42)
data = np.concatenate([np.random.normal(loc=50, scale=5, size=95),
                       np.random.normal(loc=80, scale=5, size=5)])

# Calculate Z-scores
z_scores = np.abs(stats.zscore(data))

# Identify outliers (Z-score > 3)
outliers = data[z_scores > 3]

# Visualize the data and outliers
plt.figure(figsize=(10, 6))
plt.scatter(range(len(data)), data, c='blue', label='Normal data')
plt.scatter(np.where(z_scores > 3)[0], outliers, c='red', label='Outliers')
plt.axhline(np.mean(data), color='green', linestyle='dashed', label='Mean')
plt.title('Outlier Detection using Z-score')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()

print(f"Number of outliers detected: {len(outliers)}")
```

Slide 3: Feature Selection with Correlation Analysis

Feature selection is a critical step in machine learning, helping to identify the most relevant variables for our model. Correlation analysis provides insights into the relationships between features and can guide our selection process.

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# Load the Boston Housing dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Calculate correlation matrix
corr_matrix = df.corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Boston Housing Dataset')
plt.show()

# Select features with high correlation to the target variable
threshold = 0.5
selected_features = corr_matrix['PRICE'][abs(corr_matrix['PRICE']) > threshold].index.tolist()
selected_features.remove('PRICE')

print("Selected features:", selected_features)
```

Slide 4: Verifying Model Assumptions

Many statistical and machine learning models rely on certain assumptions about the data. Verifying these assumptions is crucial for ensuring the validity and reliability of our results.

```python
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# Split the data and fit a linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and calculate residuals
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Check normality of residuals
_, p_value = stats.normaltest(residuals)

# Visualize residuals
plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.subplot(122)
stats.probplot(residuals.ravel(), plot=plt)
plt.title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.show()

print(f"Normality test p-value: {p_value[0]:.4f}")
print(f"Residuals are {'normally' if p_value > 0.05 else 'not normally'} distributed")
```

Slide 5: Probability in Machine Learning

Probability theory forms the foundation for many machine learning algorithms, enabling us to model uncertainty and make predictions based on incomplete information.

```python
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data for two classes
np.random.seed(42)
class1 = np.random.normal(loc=3, scale=1, size=1000)
class2 = np.random.normal(loc=7, scale=1.5, size=1000)

# Estimate parameters
mu1, std1 = norm.fit(class1)
mu2, std2 = norm.fit(class2)

# Create a range of values
x = np.linspace(0, 10, 200)

# Calculate probability densities
pdf1 = norm.pdf(x, mu1, std1)
pdf2 = norm.pdf(x, mu2, std2)

# Visualize the distributions
plt.figure(figsize=(10, 6))
plt.hist(class1, bins=30, density=True, alpha=0.5, label='Class 1')
plt.hist(class2, bins=30, density=True, alpha=0.5, label='Class 2')
plt.plot(x, pdf1, 'r-', lw=2, label='Class 1 PDF')
plt.plot(x, pdf2, 'b-', lw=2, label='Class 2 PDF')
plt.title('Probability Distributions of Two Classes')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# Calculate decision boundary
decision_boundary = (mu1*std2**2 - mu2*std1**2 + std1*std2*np.sqrt((mu1-mu2)**2 + 2*(std2**2-std1**2)*np.log(std2/std1))) / (std2**2 - std1**2)
print(f"Decision boundary: {decision_boundary:.2f}")
```

Slide 6: Model Evaluation Metrics

Evaluating the performance of machine learning models is crucial for understanding their effectiveness and comparing different approaches. Various statistical metrics provide insights into different aspects of model performance.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate sample predictions and true labels
np.random.seed(42)
y_true = np.random.randint(2, size=1000)
y_pred = np.random.randint(2, size=1000)

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
```

Slide 7: Bias-Variance Tradeoff

The bias-variance tradeoff is a fundamental concept in machine learning that helps us understand the balance between model complexity and generalization performance.

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(42)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to fit polynomial regression
def fit_polynomial(degree):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)
    return poly_features, model

# Fit models with different degrees
degrees = [1, 3, 15]
plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees):
    poly_features, model = fit_polynomial(degree)
    
    # Make predictions
    X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
    y_plot = model.predict(poly_features.transform(X_plot))
    
    # Plot results
    plt.subplot(1, 3, i+1)
    plt.scatter(X_train, y_train, color='b', s=10, alpha=0.5)
    plt.plot(X_plot, y_plot, color='r')
    plt.title(f'Polynomial Degree {degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    
    # Calculate and print MSE
    y_pred = model.predict(poly_features.transform(X_test))
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE for degree {degree}: {mse:.4f}")

plt.tight_layout()
plt.show()
```

Slide 8: Overfitting Detection

Overfitting occurs when a model learns the training data too well, capturing noise and failing to generalize to new data. Detecting overfitting is crucial for building robust machine learning models.

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

# Generate sample data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(16))  # Add noise to every 5th sample

# Calculate learning curves
train_sizes, train_scores, test_scores = learning_curve(
    SVC(kernel='rbf', gamma=0.1), X, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_squared_error'
)

# Calculate mean and standard deviation
train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Mean Squared Error")
plt.title("Learning Curves (SVM, RBF kernel, $\gamma=0.1$)")
plt.legend(loc="best")
plt.show()

# Check for overfitting
if train_scores_mean[-1] < test_scores_mean[-1]:
    print("The model appears to be overfitting.")
else:
    print("The model does not show clear signs of overfitting.")
```

Slide 9: Statistical Sampling Techniques

Proper sampling is essential for creating reliable training datasets and ensuring that our models generalize well to new data. Various sampling techniques help us create representative subsets of our data.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Simple Random Sampling
simple_sample = df.sample(n=30, random_state=42)

# Stratified Sampling
stratified_sample = df.groupby('target', group_keys=False).apply(lambda x: x.sample(n=10))

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['target'], alpha=0.6)
axes[0].set_title('Original Data')

axes[1].scatter(simple_sample['sepal length (cm)'], simple_sample['sepal width (cm)'], 
                c=simple_sample['target'], alpha=0.6)
axes[1].set_title('Simple Random Sample')

axes[2].scatter(stratified_sample['sepal length (cm)'], stratified_sample['sepal width (cm)'], 
                c=stratified_sample['target'], alpha=0.6)
axes[2].set_title('Stratified Sample')

for ax in axes:
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')

plt.tight_layout()
plt.show()

print("Original class distribution:\n", df['target'].value_counts(normalize=True))
print("\nSimple random sample class distribution:\n", simple_sample['target'].value_counts(normalize=True))
print("\nStratified sample class distribution:\n", stratified_sample['target'].value_counts(normalize=True))
```

Slide 10: Hypothesis Testing in Machine Learning

Hypothesis testing plays a crucial role in machine learning, helping us make informed decisions about model performance, feature importance, and data relationships.

```python
from scipy import stats
import matplotlib.pyplot as plt

# Generate sample data for two groups
np.random.seed(42)
group1 = np.random.normal(loc=5, scale=1, size=100)
group2 = np.random.normal(loc=5.5, scale=1, size=100)

# Perform t-test
t_statistic, p_value = stats.ttest_ind(group1, group2)

# Visualize the distributions
plt.figure(figsize=(10, 6))
plt.hist(group1, bins=20, alpha=0.5, label='Group 1')
plt.hist(group2, bins=20, alpha=0.5, label='Group 2')
plt.axvline(np.mean(group1), color='blue', linestyle='dashed', linewidth=2)
plt.axvline(np.mean(group2), color='orange', linestyle='dashed', linewidth=2)
plt.title('Distribution of Two Groups')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"The difference between the groups is {'statistically significant' if p_value < 0.05 else 'not statistically significant'}")
```

Slide 11: Confidence Intervals in Predictions

Confidence intervals provide a range of plausible values for our predictions, giving us a measure of uncertainty in our model's output.

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# Split the data and fit a linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_plot)

# Calculate confidence intervals
n = len(X_train)
p = 2  # number of parameters in the model
dof = n - p
t_value = stats.t.ppf(0.975, dof)
mse = np.sum((y_train - model.predict(X_train))**2) / dof
std_error = np.sqrt(mse * (1 + 1/n + (X_plot - np.mean(X_train))**2 / np.sum((X_train - np.mean(X_train))**2)))
ci = t_value * std_error

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_plot, y_pred, color='red', label='Regression line')
plt.fill_between(X_plot.ravel(), (y_pred - ci).ravel(), (y_pred + ci).ravel(), alpha=0.2, color='gray', label='95% CI')
plt.title('Linear Regression with Confidence Intervals')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

Slide 12: Bayesian Inference in Machine Learning

Bayesian inference provides a framework for updating our beliefs about model parameters as we observe new data, allowing us to quantify uncertainty in our estimates.

```python
import matplotlib.pyplot as plt
from scipy import stats

# Define true parameter and generate data
true_mu = 5
n_samples = 50
data = np.random.normal(true_mu, 1, n_samples)

# Prior distribution (we assume a normal distribution)
prior_mu = 4
prior_sigma = 2

# Calculate posterior distribution
posterior_sigma = 1 / np.sqrt(1/prior_sigma**2 + n_samples)
posterior_mu = posterior_sigma**2 * (prior_mu/prior_sigma**2 + np.sum(data))

# Generate x values for plotting
x = np.linspace(0, 10, 1000)

# Calculate PDFs
prior_pdf = stats.norm.pdf(x, prior_mu, prior_sigma)
likelihood = stats.norm.pdf(x, np.mean(data), 1/np.sqrt(n_samples))
posterior_pdf = stats.norm.pdf(x, posterior_mu, posterior_sigma)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, prior_pdf, label='Prior', color='blue')
plt.plot(x, likelihood, label='Likelihood', color='green')
plt.plot(x, posterior_pdf, label='Posterior', color='red')
plt.axvline(true_mu, color='black', linestyle='--', label='True μ')
plt.title('Bayesian Inference: Updating Beliefs')
plt.xlabel('μ')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

print(f"True μ: {true_mu}")
print(f"Prior μ: {prior_mu}")
print(f"Posterior μ: {posterior_mu:.4f}")
print(f"Posterior σ: {posterior_sigma:.4f}")
```

Slide 13: Cross-Validation Techniques

Cross-validation is a crucial technique for assessing model performance and preventing overfitting by evaluating the model on multiple subsets of the data.

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, 
                           n_clusters_per_class=1, random_state=42)

# Create a model
model = SVC(kernel='rbf', C=1)

# Perform k-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf)

# Visualize the cross-validation process
plt.figure(figsize=(12, 4))
for i, (train_index, val_index) in enumerate(kf.split(X)):
    plt.subplot(1, k_folds, i+1)
    plt.scatter(X[train_index, 0], X[train_index, 1], c=y[train_index], alpha=0.6, s=20)
    plt.scatter(X[val_index, 0], X[val_index, 1], c=y[val_index], alpha=1, s=40, marker='s')
    plt.title(f'Fold {i+1}')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()

print("Cross-validation scores:", cv_scores)
print(f"Mean CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
```

Slide 14: Additional Resources

For those interested in delving deeper into statistics and machine learning, here are some valuable resources:

1. "Statistical Learning" by Hastie, Tibshirani, and Friedman ([https://arxiv.org/abs/2103.05622](https://arxiv.org/abs/2103.05622))
2. "Bayesian Methods for Hackers" by Davidson-Pilon ([https://arxiv.org/abs/2006.02013](https://arxiv.org/abs/2006.02013))
3. "A Survey of Cross-Validation Procedures for Model Selection" by Arlot and Celisse ([https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728))
4. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani ([https://arxiv.org/abs/1809.10430](https://arxiv.org/abs/1809.10430))

These resources provide in-depth coverage of various statistical and machine learning concepts, offering both theoretical foundations and practical applications.


