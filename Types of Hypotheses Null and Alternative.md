## Types of Hypotheses Null and Alternative
Slide 1: Types of Hypotheses

Hypotheses are fundamental to scientific inquiry and statistical analysis. They serve as testable predictions about the relationships between variables. Let's explore the two main types of hypotheses used in statistical testing.

```python
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Simulating data for demonstration
np.random.seed(42)
control_group = np.random.normal(100, 15, 100)
treatment_group = np.random.normal(110, 15, 100)

# Visualizing the data
plt.figure(figsize=(10, 6))
plt.hist(control_group, alpha=0.5, label='Control Group')
plt.hist(treatment_group, alpha=0.5, label='Treatment Group')
plt.legend()
plt.title('Distribution of Control and Treatment Groups')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: Null Hypothesis (H0)

The null hypothesis (H0) is the default assumption that there is no effect or no difference between groups. It's the starting point for statistical testing, representing the status quo or the absence of an effect.

```python
def null_hypothesis_example():
    """
    Example of a null hypothesis: There is no significant difference 
    in exam scores between students who studied for 2 hours and 4 hours.
    """
    two_hour_scores = np.random.normal(75, 10, 50)
    four_hour_scores = np.random.normal(78, 10, 50)
    
    t_statistic, p_value = stats.ttest_ind(two_hour_scores, four_hour_scores)
    
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")
    print(f"Null hypothesis {'rejected' if p_value < 0.05 else 'not rejected'}")

null_hypothesis_example()
```

Slide 3: Alternative Hypothesis (H1 or Ha)

The alternative hypothesis (H1 or Ha) is what researchers suspect might be true instead of the null hypothesis. It represents the presence of an effect or a significant difference between groups.

```python
def alternative_hypothesis_example():
    """
    Example of an alternative hypothesis: Students who studied for 4 hours 
    have significantly higher exam scores than those who studied for 2 hours.
    """
    two_hour_scores = np.random.normal(75, 10, 50)
    four_hour_scores = np.random.normal(85, 10, 50)
    
    t_statistic, p_value = stats.ttest_ind(two_hour_scores, four_hour_scores)
    
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")
    print(f"Alternative hypothesis {'supported' if p_value < 0.05 else 'not supported'}")

alternative_hypothesis_example()
```

Slide 4: Steps in Hypothesis Testing - State the Hypotheses

The first step in hypothesis testing is to clearly state both the null and alternative hypotheses. This sets the foundation for the entire testing process.

```python
def state_hypotheses():
    """
    Example of stating hypotheses for a study on the effect of a new fertilizer on plant growth.
    """
    null_hypothesis = "There is no significant difference in plant height between the control and fertilizer groups."
    alternative_hypothesis = "Plants treated with the new fertilizer have significantly different heights compared to the control group."
    
    print("Null Hypothesis (H0):", null_hypothesis)
    print("Alternative Hypothesis (H1):", alternative_hypothesis)

state_hypotheses()
```

Slide 5: Choose a Significance Level (α)

The significance level (α) is the probability of rejecting the null hypothesis when it is actually true (Type I error). Common levels include 5% (0.05) and 1% (0.01).

```python
def choose_significance_level():
    """
    Demonstrating the choice of significance level and its implications.
    """
    alpha_values = [0.1, 0.05, 0.01]
    
    for alpha in alpha_values:
        print(f"Significance level (α): {alpha}")
        print(f"Confidence level: {(1 - alpha) * 100}%")
        print(f"Chance of Type I error: {alpha * 100}%")
        print()

choose_significance_level()
```

Slide 6: Collect Data & Calculate Test Statistic

After stating the hypotheses and choosing a significance level, the next step is to collect data and calculate a test statistic. The type of test statistic depends on the nature of your data and hypothesis.

```python
def calculate_test_statistic():
    """
    Example of calculating a t-statistic for independent samples.
    """
    group1 = np.random.normal(100, 15, 30)
    group2 = np.random.normal(110, 15, 30)
    
    t_statistic, _ = stats.ttest_ind(group1, group2)
    
    print(f"Group 1 mean: {np.mean(group1):.2f}")
    print(f"Group 2 mean: {np.mean(group2):.2f}")
    print(f"T-statistic: {t_statistic:.4f}")

calculate_test_statistic()
```

Slide 7: Determine the p-value

The p-value is the probability of observing your results (or more extreme) if the null hypothesis is true. It's a crucial component in making decisions about the hypotheses.

```python
def determine_p_value():
    """
    Example of calculating and interpreting p-values.
    """
    group1 = np.random.normal(100, 15, 30)
    group2 = np.random.normal(110, 15, 30)
    
    _, p_value = stats.ttest_ind(group1, group2)
    
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("The result is statistically significant at the 0.05 level.")
    else:
        print("The result is not statistically significant at the 0.05 level.")

determine_p_value()
```

Slide 8: Make a Decision

Based on the p-value and the chosen significance level, you make a decision to either reject or fail to reject the null hypothesis. This decision guides the interpretation of your results.

```python
def make_decision(p_value, alpha):
    """
    Function to make a decision based on p-value and significance level.
    """
    if p_value < alpha:
        return "Reject the null hypothesis"
    else:
        return "Fail to reject the null hypothesis"

# Example usage
p_value = 0.03
alpha = 0.05

decision = make_decision(p_value, alpha)
print(f"P-value: {p_value}")
print(f"Significance level (α): {alpha}")
print(f"Decision: {decision}")
```

Slide 9: Why is Hypothesis Testing Important?

Hypothesis testing provides a structured approach to making decisions about populations based on sample data. It's a cornerstone of scientific research and data-driven decision-making across various fields.

```python
def importance_of_hypothesis_testing():
    """
    Simulating a scenario to demonstrate the importance of hypothesis testing.
    """
    np.random.seed(42)
    
    # Simulating two marketing strategies
    strategy_a = np.random.binomial(1, 0.1, 1000)  # 10% conversion rate
    strategy_b = np.random.binomial(1, 0.12, 1000)  # 12% conversion rate
    
    # Performing hypothesis test
    _, p_value = stats.ttest_ind(strategy_a, strategy_b)
    
    print(f"Strategy A conversion rate: {np.mean(strategy_a):.2%}")
    print(f"Strategy B conversion rate: {np.mean(strategy_b):.2%}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("There is a significant difference between the strategies.")
    else:
        print("There is no significant difference between the strategies.")

importance_of_hypothesis_testing()
```

Slide 10: Applications in Real Life - Healthcare

In healthcare, hypothesis testing is crucial for evaluating the effectiveness of new treatments. Let's simulate a scenario comparing a new treatment against a placebo.

```python
def healthcare_example():
    """
    Simulating a clinical trial comparing a new treatment to a placebo.
    """
    np.random.seed(42)
    
    placebo = np.random.normal(5, 2, 100)  # Symptom reduction with placebo
    treatment = np.random.normal(6, 2, 100)  # Symptom reduction with new treatment
    
    t_statistic, p_value = stats.ttest_ind(placebo, treatment)
    
    print(f"Average symptom reduction (placebo): {np.mean(placebo):.2f}")
    print(f"Average symptom reduction (treatment): {np.mean(treatment):.2f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("The new treatment shows a significant effect compared to the placebo.")
    else:
        print("There is no significant difference between the new treatment and the placebo.")

healthcare_example()
```

Slide 11: Applications in Real Life - Manufacturing

In manufacturing, hypothesis testing can be used to compare the quality of products produced by different methods. Let's simulate a scenario comparing two production methods.

```python
def manufacturing_example():
    """
    Simulating a comparison of two manufacturing methods.
    """
    np.random.seed(42)
    
    old_method = np.random.normal(100, 5, 100)  # Product quality scores
    new_method = np.random.normal(102, 5, 100)  # Product quality scores
    
    t_statistic, p_value = stats.ttest_ind(old_method, new_method)
    
    print(f"Average quality score (old method): {np.mean(old_method):.2f}")
    print(f"Average quality score (new method): {np.mean(new_method):.2f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("The new manufacturing method produces significantly different quality products.")
    else:
        print("There is no significant difference in quality between the two manufacturing methods.")

manufacturing_example()
```

Slide 12: Type I and Type II Errors

In hypothesis testing, it's important to understand the potential errors. Type I error occurs when we reject a true null hypothesis, while Type II error occurs when we fail to reject a false null hypothesis.

```python
def error_types_simulation(n_simulations=10000, alpha=0.05):
    """
    Simulating Type I and Type II errors.
    """
    # True null hypothesis scenario
    null_true = np.random.normal(0, 1, (n_simulations, 100))
    
    # False null hypothesis scenario
    null_false = np.random.normal(0.5, 1, (n_simulations, 100))
    
    # Calculating p-values
    _, p_values_null_true = stats.ttest_1samp(null_true, 0, axis=1)
    _, p_values_null_false = stats.ttest_1samp(null_false, 0, axis=1)
    
    # Calculating error rates
    type_i_error = np.mean(p_values_null_true < alpha)
    type_ii_error = np.mean(p_values_null_false >= alpha)
    
    print(f"Simulated Type I error rate: {type_i_error:.4f}")
    print(f"Simulated Type II error rate: {type_ii_error:.4f}")

error_types_simulation()
```

Slide 13: Power Analysis

Power analysis helps determine the sample size needed to detect an effect of a given size with a certain level of confidence. It's crucial for designing robust experiments.

```python
from statsmodels.stats.power import TTestIndPower

def power_analysis_example():
    """
    Demonstrating power analysis for a two-sample t-test.
    """
    effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large effect sizes
    alpha = 0.05
    power = 0.8
    
    power_analysis = TTestIndPower()
    
    for effect_size in effect_sizes:
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=1.0,
            alternative='two-sided'
        )
        print(f"For effect size {effect_size}, required sample size: {sample_size:.0f}")

power_analysis_example()
```

Slide 14: Limitations and Considerations

While hypothesis testing is a powerful tool, it's important to understand its limitations and use it appropriately. Factors like sample size, effect size, and practical significance should be considered alongside statistical significance.

```python
def limitations_example():
    """
    Demonstrating how a large sample size can lead to statistically
    significant results that may not be practically significant.
    """
    np.random.seed(42)
    
    # Simulating two very large groups with a tiny difference
    group1 = np.random.normal(100, 10, 10000)
    group2 = np.random.normal(100.1, 10, 10000)
    
    t_statistic, p_value = stats.ttest_ind(group1, group2)
    
    print(f"Group 1 mean: {np.mean(group1):.2f}")
    print(f"Group 2 mean: {np.mean(group2):.2f}")
    print(f"Difference in means: {np.mean(group2) - np.mean(group1):.2f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("The result is statistically significant, but is it practically significant?")

limitations_example()
```

Slide 15: Additional Resources

For those interested in delving deeper into hypothesis testing and statistical analysis, here are some recommended resources:

1. "Statistical Inference via Data Science: A ModernDive into R and the Tidyverse" by Chester Ismay and Albert Y. Kim (available at [https://moderndive.com/](https://moderndive.com/))
2. "Introduction to Statistical Learning" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (available at [https://www.statlearning.com/](https://www.statlearning.com/))
3. "Statistical Rethinking: A Bayesian Course with Examples in R and Stan" by Richard McElreath (arXiv:2011.01808)

These resources provide comprehensive coverage of statistical concepts and their practical applications in data science and research.

