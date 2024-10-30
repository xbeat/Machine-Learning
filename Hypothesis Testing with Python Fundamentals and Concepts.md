## Hypothesis Testing with Python Fundamentals and Concepts
Slide 1: Hypothesis Testing Fundamentals

Statistical hypothesis testing provides a framework for making decisions about populations based on sample data. It involves formulating null and alternative hypotheses, choosing a significance level, calculating test statistics, and making conclusions based on probability values.

```python
# Basic structure of hypothesis testing
import numpy as np
from scipy import stats

def hypothesis_test(sample_data, population_mean, alpha=0.05):
    # Calculate test statistic
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)
    n = len(sample_data)
    t_stat = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
    
    # Calculate p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    # Make decision
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'reject_null': p_value < alpha
    }
```

Slide 2: One-Sample T-Test Implementation

The one-sample t-test determines whether a sample mean significantly differs from a hypothesized population mean. This implementation demonstrates the complete process including assumption checking, test statistic calculation, and result interpretation.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def one_sample_ttest(data, mu0, alpha=0.05):
    # Check normality assumption
    _, normality_p = stats.normaltest(data)
    
    # Perform t-test
    t_stat, p_value = stats.ttest_1samp(data, mu0)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.7)
    plt.axvline(mu0, color='r', linestyle='--', label='Null hypothesis mean')
    plt.title('Sample Distribution vs Null Hypothesis')
    plt.legend()
    
    return {
        'normality_p': normality_p,
        't_statistic': t_stat,
        'p_value': p_value,
        'reject_null': p_value < alpha
    }
```

Slide 3: Two-Sample T-Test Analysis

This test compares means of two independent samples to determine if they differ significantly. The implementation includes both equal and unequal variance cases, along with effect size calculation using Cohen's d.

```python
def two_sample_ttest(group1, group2, alpha=0.05):
    # Test for equal variances
    _, levene_p = stats.levene(group1, group2)
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, 
                                     equal_var=(levene_p > alpha))
    
    # Calculate Cohen's d
    pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return {
        'equal_variance': levene_p > alpha,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'reject_null': p_value < alpha
    }
```

Slide 4: Paired T-Test Implementation

The paired t-test analyzes differences between paired observations. This implementation handles paired data analysis, including correlation assessment and visualization of differences between pairs.

```python
def paired_ttest(before, after, alpha=0.05):
    # Calculate differences
    differences = after - before
    
    # Test normality of differences
    _, norm_p = stats.normaltest(differences)
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(before, after)
    
    # Calculate effect size
    cohen_d = np.mean(differences) / np.std(differences)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(before, after, alpha=0.5)
    plt.plot([min(before), max(before)], [min(before), max(before)], 
             'r--', label='No change line')
    plt.xlabel('Before')
    plt.ylabel('After')
    plt.title('Paired Data Visualization')
    plt.legend()
    
    return {
        'normality_p': norm_p,
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': cohen_d,
        'reject_null': p_value < alpha
    }
```

Slide 5: Power Analysis and Sample Size Calculation

Power analysis determines the minimum sample size needed to detect an effect of a given size with specified significance level and power. This implementation provides functions for calculating power and required sample size for various test types.

```python
from scipy import stats
import numpy as np

def power_analysis(effect_size, alpha=0.05, power=0.8, test_type='two_sample'):
    if test_type == 'two_sample':
        # Calculate required sample size for two-sample t-test
        n = stats.tt_ind_solve_power(effect_size=effect_size, 
                                   alpha=alpha, 
                                   power=power, 
                                   ratio=1.0, 
                                   alternative='two-sided')
    else:
        # Calculate required sample size for one-sample t-test
        n = stats.tt_solve_power(effect_size=effect_size, 
                               alpha=alpha, 
                               power=power, 
                               alternative='two-sided')
    
    return {
        'required_sample_size': np.ceil(n),
        'effect_size': effect_size,
        'alpha': alpha,
        'power': power
    }
```

Slide 6: Multiple Hypothesis Testing Correction

When performing multiple hypothesis tests simultaneously, the probability of Type I errors increases. This implementation demonstrates various methods for p-value adjustment, including Bonferroni and False Discovery Rate corrections.

```python
def multiple_testing_correction(p_values, method='bonferroni'):
    """
    Implements multiple testing correction methods
    """
    n_tests = len(p_values)
    
    if method == 'bonferroni':
        # Bonferroni correction
        adjusted_p = np.minimum(p_values * n_tests, 1.0)
    
    elif method == 'fdr':
        # Benjamini-Hochberg FDR
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Calculate FDR adjusted p-values
        adjusted_p = np.zeros_like(p_values)
        for i, p in enumerate(sorted_p):
            adjusted_p[sorted_idx[i]] = p * n_tests / (i + 1)
        
        # Ensure monotonicity
        for i in range(len(adjusted_p)-2, -1, -1):
            adjusted_p[i] = min(adjusted_p[i], adjusted_p[i+1])
    
    return adjusted_p
```

Slide 7: ANOVA Implementation

Analysis of Variance (ANOVA) tests differences among group means in a sample. This implementation includes one-way ANOVA with post-hoc tests and effect size calculations using eta-squared.

```python
def one_way_anova(groups, alpha=0.05):
    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Calculate eta-squared
    ss_between = sum(len(g) * (np.mean(g) - np.mean(np.concatenate(groups)))**2 
                    for g in groups)
    ss_total = sum(sum((x - np.mean(np.concatenate(groups)))**2) 
                  for g in groups)
    eta_squared = ss_between / ss_total
    
    # Post-hoc Tukey HSD if ANOVA is significant
    post_hoc = None
    if p_value < alpha:
        data = np.concatenate(groups)
        groups_idx = np.repeat(range(len(groups)), 
                             [len(g) for g in groups])
        post_hoc = stats.tukey_hsd(data, groups_idx)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'post_hoc': post_hoc,
        'reject_null': p_value < alpha
    }
```

Slide 8: Real-World Example - Clinical Trial Analysis

This example demonstrates a complete analysis of a clinical trial comparing treatment effectiveness. It includes data preprocessing, statistical testing, and comprehensive result interpretation for medical research.

```python
import pandas as pd
import numpy as np
from scipy import stats

def clinical_trial_analysis(treatment_group, control_group):
    # Data preprocessing
    treatment = np.array(treatment_group)
    control = np.array(control_group)
    
    # Descriptive statistics
    stats_summary = {
        'treatment': {
            'mean': np.mean(treatment),
            'std': np.std(treatment),
            'n': len(treatment)
        },
        'control': {
            'mean': np.mean(control),
            'std': np.std(control),
            'n': len(control)
        }
    }
    
    # Perform statistical tests
    # 1. Check normality
    _, norm_p_treat = stats.normaltest(treatment)
    _, norm_p_ctrl = stats.normaltest(control)
    
    # 2. Two-sample t-test
    t_stat, p_value = stats.ttest_ind(treatment, control)
    
    # 3. Effect size calculation
    pooled_std = np.sqrt(((len(treatment)-1) * np.var(treatment) + 
                         (len(control)-1) * np.var(control)) / 
                        (len(treatment) + len(control) - 2))
    cohen_d = (np.mean(treatment) - np.mean(control)) / pooled_std
    
    return {
        'summary': stats_summary,
        'normality': {'treatment_p': norm_p_treat, 'control_p': norm_p_ctrl},
        't_test': {'statistic': t_stat, 'p_value': p_value},
        'effect_size': cohen_d
    }
```

Slide 9: Results Visualization and Reporting

Statistical analysis results require clear and informative visualization. This implementation creates comprehensive statistical reports with interactive visualizations using seaborn and matplotlib for hypothesis testing outcomes.

```python
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_test_results(data_groups, test_results, test_type='t_test'):
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution plots
    for group_name, group_data in data_groups.items():
        sns.kdeplot(data=group_data, ax=ax1, label=group_name)
    ax1.set_title('Distribution Comparison')
    ax1.legend()
    
    # Box plot
    sns.boxplot(data=list(data_groups.values()), ax=ax2)
    ax2.set_xticklabels(data_groups.keys())
    ax2.set_title('Box Plot Comparison')
    
    # QQ Plot for normality check
    for group_name, group_data in data_groups.items():
        stats.probplot(group_data, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    
    # Effect size visualization
    if 'effect_size' in test_results:
        effect_sizes = [test_results['effect_size']]
        ax4.bar(['Cohen\'s d'], effect_sizes)
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax4.set_title('Effect Size')
    
    plt.tight_layout()
    return fig
```

Slide 10: Real-World Example - Market Research Analysis

Applying hypothesis testing to market research data, comparing customer satisfaction scores across different product lines with comprehensive statistical analysis and visualization.

```python
def market_research_analysis(satisfaction_data):
    """
    satisfaction_data: dict with product lines as keys and satisfaction scores as values
    """
    # Data preparation
    product_lines = list(satisfaction_data.keys())
    scores = list(satisfaction_data.values())
    
    # Descriptive statistics
    desc_stats = {prod: {
        'mean': np.mean(scores[i]),
        'median': np.median(scores[i]),
        'std': np.std(scores[i]),
        'n': len(scores[i])
    } for i, prod in enumerate(product_lines)}
    
    # ANOVA test
    f_stat, p_value = stats.f_oneway(*scores)
    
    # Effect size calculation (eta-squared)
    df_between = len(product_lines) - 1
    df_total = sum(len(s) for s in scores) - 1
    eta_squared = (df_between * f_stat) / (df_between * f_stat + df_total)
    
    # Post-hoc analysis if ANOVA is significant
    tukey_results = None
    if p_value < 0.05:
        all_data = np.concatenate(scores)
        groups = np.repeat(range(len(scores)), [len(s) for s in scores])
        tukey_results = stats.tukey_hsd(all_data, groups)
    
    return {
        'descriptive': desc_stats,
        'anova': {'f_stat': f_stat, 'p_value': p_value},
        'effect_size': eta_squared,
        'post_hoc': tukey_results
    }
```

Slide 11: Bootstrap Hypothesis Testing

Bootstrap methods provide robust hypothesis testing when parametric assumptions are violated. This implementation demonstrates resampling-based hypothesis testing with confidence interval calculation.

```python
def bootstrap_test(sample1, sample2, n_bootstrap=10000, alpha=0.05):
    # Calculate observed difference in means
    observed_diff = np.mean(sample1) - np.mean(sample2)
    
    # Combined sample for null hypothesis
    combined = np.concatenate([sample1, sample2])
    n1, n2 = len(sample1), len(sample2)
    
    # Bootstrap resampling
    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample under null hypothesis
        resampled = np.random.choice(combined, size=n1+n2, replace=True)
        boot_sample1 = resampled[:n1]
        boot_sample2 = resampled[n1:]
        bootstrap_diffs[i] = np.mean(boot_sample1) - np.mean(boot_sample2)
    
    # Calculate p-value
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    
    # Calculate confidence interval
    ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1-alpha/2) * 100)
    
    return {
        'observed_difference': observed_diff,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'reject_null': p_value < alpha
    }
```

Slide 12: Non-Parametric Hypothesis Testing

Non-parametric tests make fewer assumptions about data distribution. This implementation includes Mann-Whitney U test, Wilcoxon signed-rank test, and Kruskal-Wallis H test for various experimental designs.

```python
def nonparametric_analysis(data_groups, test_type='mann_whitney'):
    """
    Comprehensive non-parametric testing suite
    """
    results = {}
    
    if test_type == 'mann_whitney':
        # Mann-Whitney U test for two independent samples
        stat, p_value = stats.mannwhitneyu(
            data_groups[0], 
            data_groups[1],
            alternative='two-sided'
        )
        results['test_name'] = 'Mann-Whitney U'
        results['statistic'] = stat
        
    elif test_type == 'wilcoxon':
        # Wilcoxon signed-rank test for paired samples
        stat, p_value = stats.wilcoxon(
            data_groups[0],
            data_groups[1],
            alternative='two-sided'
        )
        results['test_name'] = 'Wilcoxon Signed-Rank'
        results['statistic'] = stat
        
    elif test_type == 'kruskal':
        # Kruskal-Wallis H test for multiple groups
        stat, p_value = stats.kruskal(*data_groups)
        results['test_name'] = 'Kruskal-Wallis H'
        results['statistic'] = stat
    
    # Calculate effect size (r for Mann-Whitney U and Wilcoxon)
    if test_type in ['mann_whitney', 'wilcoxon']:
        n1, n2 = len(data_groups[0]), len(data_groups[1])
        results['effect_size'] = abs(stat) / np.sqrt(n1 * n2)
    
    results['p_value'] = p_value
    results['reject_null'] = p_value < 0.05
    
    return results
```

Slide 13: Cross-Validation for Hypothesis Testing

Cross-validation techniques enhance the reliability of hypothesis testing results. This implementation demonstrates k-fold cross-validated hypothesis testing with stability assessment.

```python
def cross_validated_hypothesis_test(data1, data2, n_folds=5, test_func=stats.ttest_ind):
    """
    Performs cross-validated hypothesis testing
    """
    # Create fold indices
    n1, n2 = len(data1), len(data2)
    fold_size1 = n1 // n_folds
    fold_size2 = n2 // n_folds
    
    results = []
    for i in range(n_folds):
        # Create test-train splits
        test_idx1 = slice(i * fold_size1, (i + 1) * fold_size1)
        test_idx2 = slice(i * fold_size2, (i + 1) * fold_size2)
        
        # Perform test on fold
        stat, p_val = test_func(
            data1[test_idx1],
            data2[test_idx2]
        )
        
        results.append({
            'fold': i + 1,
            'statistic': stat,
            'p_value': p_val
        })
    
    # Calculate stability metrics
    p_values = [r['p_value'] for r in results]
    stats = [r['statistic'] for r in results]
    
    return {
        'fold_results': results,
        'mean_p_value': np.mean(p_values),
        'std_p_value': np.std(p_values),
        'mean_statistic': np.mean(stats),
        'std_statistic': np.std(stats),
        'stable_significance': all(p < 0.05 for p in p_values)
    }
```

Slide 14: Performance Analysis and Effect Sizes

A comprehensive suite for calculating and interpreting different effect size measures across various statistical tests, including Cohen's d, Hedges' g, and Glass's delta.

```python
def effect_size_analysis(sample1, sample2, test_type='cohens_d'):
    """
    Calculates various effect size measures
    """
    results = {}
    
    # Basic statistics
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    
    if test_type == 'cohens_d':
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / 
                           (n1 + n2 - 2))
        effect_size = (mean1 - mean2) / pooled_std
        
    elif test_type == 'hedges_g':
        # Hedges' g (bias-corrected)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / 
                           (n1 + n2 - 2))
        correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
        effect_size = correction * (mean1 - mean2) / pooled_std
        
    elif test_type == 'glass_delta':
        # Glass's delta (using control group std)
        effect_size = (mean1 - mean2) / np.sqrt(var2)
    
    # Calculate confidence intervals
    se = np.sqrt((n1 + n2) / (n1 * n2) + effect_size**2 / (2*(n1 + n2)))
    ci_lower = effect_size - 1.96 * se
    ci_upper = effect_size + 1.96 * se
    
    return {
        'effect_size': effect_size,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'interpretation': interpret_effect_size(effect_size, test_type)
    }

def interpret_effect_size(effect_size, test_type):
    """
    Provides standardized interpretation of effect sizes
    """
    abs_effect = abs(effect_size)
    
    if test_type in ['cohens_d', 'hedges_g']:
        if abs_effect < 0.2:
            return 'negligible'
        elif abs_effect < 0.5:
            return 'small'
        elif abs_effect < 0.8:
            return 'medium'
        else:
            return 'large'
    
    return 'custom interpretation needed'
```

Slide 15: Additional Resources

*   "A New Look at the Statistical Model Identification" [https://arxiv.org/abs/1404.2000](https://arxiv.org/abs/1404.2000)
*   "Bootstrap Methods: Another Look at the Jackknife" [https://arxiv.org/abs/1208.4118](https://arxiv.org/abs/1208.4118)
*   "False Discovery Rate Control: A Practical and Powerful Approach" [https://arxiv.org/abs/1309.2848](https://arxiv.org/abs/1309.2848)
*   "The Problem of Multiple Comparisons in Statistics" [https://arxiv.org/abs/1411.5786](https://arxiv.org/abs/1411.5786)
*   "Effect Size Measures for Mediation Models" [https://arxiv.org/abs/1507.06665](https://arxiv.org/abs/1507.06665)

