## Raincloud Plots Revealing Hidden Data Distributions
Slide 1: Understanding Data Distribution Visualization Challenges

Traditional visualization methods like histograms and box plots can mask important patterns in data distributions. Box plots may show identical statistics for drastically different distributions, while histograms are highly sensitive to bin selection, potentially obscuring underlying patterns.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate two different distributions with similar statistics
np.random.seed(42)
dist1 = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(0, 1, 500)])
dist2 = np.concatenate([np.random.normal(-2, 0.5, 500), np.random.normal(2, 0.5, 500)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot boxplots to show similar statistics
sns.boxplot(data=[dist1, dist2], ax=ax1)
ax1.set_xticklabels(['Distribution 1', 'Distribution 2'])
ax1.set_title('Box Plots Masking Different Distributions')

# Plot histograms to reveal true distributions
sns.histplot(dist1, ax=ax2, label='Distribution 1', alpha=0.5)
sns.histplot(dist2, ax=ax2, label='Distribution 2', alpha=0.5)
ax2.set_title('True Underlying Distributions')
ax2.legend()

plt.tight_layout()
plt.show()
```

Slide 2: Introduction to Raincloud Plots

Raincloud plots combine box plots, kernel density estimation (KDE), and individual data points into a single visualization. This powerful combination provides a complete picture of data distribution, statistical summaries, and raw data points simultaneously.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_basic_raincloud(data, ax, color):
    # Create violin plot (KDE)
    sns.violinplot(data=data, ax=ax, color=color, alpha=0.3)
    
    # Add box plot
    sns.boxplot(data=data, ax=ax, width=0.1, color='white', 
                showfliers=False, zorder=2)
    
    # Add strip plot (individual points)
    sns.stripplot(data=data, ax=ax, color=color, alpha=0.4, 
                 jitter=0.1, zorder=1)

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 200)

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))
create_basic_raincloud(data, ax, 'blue')
ax.set_title('Basic Raincloud Plot')
plt.show()
```

Slide 3: Implementing Custom Raincloud Plot Function

A robust implementation of raincloud plots requires careful consideration of layout and styling. This custom function provides a flexible foundation for creating publication-quality raincloud plots with adjustable parameters.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def raincloud_plot(data, ax, color, width_viol=0.7, width_box=0.15,
                  point_size=4, alpha=0.6):
    """
    Create a raincloud plot combining KDE, boxplot, and strip plot.
    
    Parameters:
    -----------
    data : array-like
        Input data
    ax : matplotlib axis
        Axis to plot on
    color : str
        Color for the plot
    width_viol : float
        Width of violin plot
    width_box : float
        Width of box plot
    point_size : int
        Size of individual points
    alpha : float
        Transparency level
    """
    
    # Calculate kernel density
    density = stats.gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 200)
    ys = density(xs)
    
    # Plot the KDE
    ax.fill_betweenx(xs, ys, alpha=alpha, color=color)
    
    # Add boxplot
    bp = ax.boxplot(data, positions=[0], vert=False, widths=[width_box],
                    patch_artist=True, showfliers=False)
    plt.setp(bp['boxes'], facecolor='white', alpha=1.0)
    plt.setp(bp['medians'], color='black')
    
    # Add strip plot
    ax.scatter(data, np.zeros_like(data) - width_viol/2,
              alpha=alpha, s=point_size, color=color)
    
    # Cleanup
    ax.set_ylim(min(xs), max(xs))
    ax.set_yticks([])
    
# Example usage
np.random.seed(42)
data = np.random.normal(0, 1, 200)

fig, ax = plt.subplots(figsize=(8, 6))
raincloud_plot(data, ax, 'purple')
ax.set_title('Custom Raincloud Plot')
plt.show()
```

Slide 4: Comparing Multiple Groups with Raincloud Plots

When comparing multiple groups, raincloud plots excel at revealing differences in distribution shapes, central tendencies, and data spread simultaneously. This implementation handles multiple groups with automatic positioning and color coding.

```python
def multi_raincloud_plot(data_dict, ax, colors=None, spacing=0.3):
    """
    Create raincloud plots for multiple groups.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of group names and their data
    ax : matplotlib axis
        Axis to plot on
    colors : list
        List of colors for each group
    spacing : float
        Vertical spacing between groups
    """
    if colors is None:
        colors = plt.cm.Set2(np.linspace(0, 1, len(data_dict)))
    
    for idx, (group, data) in enumerate(data_dict.items()):
        pos = idx * spacing
        raincloud_plot(data, ax, colors[idx])
        ax.text(-4, pos, group, ha='right', va='center')
    
    ax.set_ylim(-0.5, len(data_dict) * spacing - 0.5)

# Generate example data
np.random.seed(42)
data = {
    'Group A': np.random.normal(0, 1, 200),
    'Group B': np.random.gamma(2, 1.5, 200),
    'Group C': np.random.beta(5, 2, 200) * 10
}

fig, ax = plt.subplots(figsize=(10, 8))
multi_raincloud_plot(data, ax)
ax.set_title('Multiple Group Comparison with Raincloud Plots')
plt.show()
```

Slide 5: Adding Statistical Annotations to Raincloud Plots

Incorporating statistical information enhances the interpretability of raincloud plots. This implementation adds significance markers and numerical statistics, making it suitable for scientific publications and detailed data analysis.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def raincloud_plot_with_stats(data_dict, ax, test='mann_whitney'):
    """
    Create raincloud plots with statistical annotations.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing group data
    ax : matplotlib axis
        Axis to plot on
    test : str
        Statistical test to perform ('mann_whitney' or 't_test')
    """
    positions = np.arange(len(data_dict))
    colors = plt.cm.Set2(np.linspace(0, 1, len(data_dict)))
    
    # Plot rainclouds
    for idx, (group, data) in enumerate(data_dict.items()):
        # KDE
        sns.kdeplot(data=data, ax=ax, color=colors[idx], fill=True, alpha=0.3)
        # Box plot
        sns.boxplot(data=data, ax=ax, color='white', width=0.1, 
                   positions=[positions[idx]])
        # Strip plot
        sns.stripplot(data=data, ax=ax, color=colors[idx], alpha=0.4, 
                     size=4, jitter=0.1)
        
        # Add statistics
        mean = np.mean(data)
        std = np.std(data)
        ax.text(positions[idx], ax.get_ylim()[1], 
                f'μ={mean:.2f}\nσ={std:.2f}',
                ha='center', va='bottom')
    
    # Perform statistical tests between groups
    if len(data_dict) > 1:
        for i in range(len(data_dict)-1):
            for j in range(i+1, len(data_dict)):
                if test == 'mann_whitney':
                    stat, p_val = stats.mannwhitneyu(
                        list(data_dict.values())[i],
                        list(data_dict.values())[j]
                    )
                else:
                    stat, p_val = stats.ttest_ind(
                        list(data_dict.values())[i],
                        list(data_dict.values())[j]
                    )
                
                # Add significance markers
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                
                y = ax.get_ylim()[1] * 1.1
                x1, x2 = positions[i], positions[j]
                ax.plot([x1, x2], [y, y], '-k')
                ax.text((x1 + x2) / 2, y, sig, ha='center', va='bottom')

# Example usage
np.random.seed(42)
example_data = {
    'Control': np.random.normal(0, 1, 100),
    'Treatment A': np.random.normal(0.5, 1.2, 100),
    'Treatment B': np.random.normal(1.5, 0.8, 100)
}

fig, ax = plt.subplots(figsize=(12, 6))
raincloud_plot_with_stats(example_data, ax)
ax.set_title('Raincloud Plot with Statistical Annotations')
plt.show()
```

Slide 6: Handling Time Series Data with Raincloud Plots

Raincloud plots can effectively visualize temporal patterns in data distributions. This implementation shows how to create time-series raincloud plots with proper temporal alignment and transitions.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def temporal_raincloud(time_series_data, time_points, ax):
    """
    Create raincloud plots for time series data.
    
    Parameters:
    -----------
    time_series_data : dict
        Dictionary with time points as keys and data arrays as values
    time_points : list
        List of time points to plot
    ax : matplotlib axis
        Axis to plot on
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))
    
    for idx, (time, data) in enumerate(time_series_data.items()):
        position = idx
        
        # KDE plot
        sns.kdeplot(data=data, ax=ax, color=colors[idx], fill=True,
                   alpha=0.3, label=f'T{time}')
        
        # Box plot
        bp = ax.boxplot(data, positions=[position], widths=0.1,
                       patch_artist=True, showfliers=False)
        plt.setp(bp['boxes'], facecolor='white')
        
        # Strip plot
        y = np.random.normal(position, 0.04, size=len(data))
        ax.scatter(data, y, alpha=0.4, s=3, color=colors[idx])
    
    ax.set_yticks(range(len(time_points)))
    ax.set_yticklabels([f'T{t}' for t in time_points])
    ax.set_ylabel('Time Points')
    ax.set_xlabel('Values')

# Generate example time series data
np.random.seed(42)
time_points = [0, 1, 2, 3]
time_series_data = {
    t: np.random.normal(t/2, 1 + t/4, 100) for t in time_points
}

fig, ax = plt.subplots(figsize=(10, 8))
temporal_raincloud(time_series_data, time_points, ax)
ax.set_title('Temporal Raincloud Plot')
plt.legend()
plt.show()
```

Slide 7: Real-world Example: Clinical Trial Data Analysis

Analyzing clinical trial data requires careful visualization of treatment effects and patient responses. This implementation demonstrates how to use raincloud plots for comprehensive clinical data visualization.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def clinical_raincloud(data, outcome_var, group_var, time_var, ax):
    """
    Create raincloud plots for clinical trial data.
    
    Parameters:
    -----------
    data : pandas DataFrame
        Clinical trial data
    outcome_var : str
        Name of outcome variable
    group_var : str
        Name of treatment group variable
    time_var : str
        Name of time point variable
    ax : matplotlib axis
        Axis to plot on
    """
    groups = data[group_var].unique()
    times = data[time_var].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    
    for t_idx, time in enumerate(times):
        for g_idx, group in enumerate(groups):
            pos = t_idx + g_idx * 0.3
            
            # Get data for this group and time
            subset = data[
                (data[group_var] == group) & 
                (data[time_var] == time)
            ][outcome_var]
            
            # Create raincloud components
            violin = ax.violinplot(subset, positions=[pos],
                                 showmeans=False, showextrema=False)
            for pc in violin['bodies']:
                pc.set_facecolor(colors[g_idx])
                pc.set_alpha(0.3)
            
            # Add box plot
            bp = ax.boxplot(subset, positions=[pos], widths=0.1,
                          showfliers=False, patch_artist=True)
            plt.setp(bp['boxes'], facecolor='white')
            
            # Add strip plot
            y = np.random.normal(pos, 0.02, size=len(subset))
            ax.scatter(subset, y, alpha=0.4, s=3, color=colors[g_idx])
            
            # Add statistics
            mean = np.mean(subset)
            ci = stats.t.interval(0.95, len(subset)-1,
                                loc=mean,
                                scale=stats.sem(subset))
            ax.plot([ci[0], ci[1]], [pos, pos], color='black', linewidth=2)

# Generate example clinical trial data
np.random.seed(42)
n_patients = 100
times = [0, 4, 8, 12]  # weeks
groups = ['Placebo', 'Treatment']

data = []
for group in groups:
    baseline = np.random.normal(100, 10, n_patients)
    effect = 0 if group == 'Placebo' else 20
    
    for time in times:
        response = baseline + effect * (time/12) + \
                  np.random.normal(0, 5, n_patients)
        data.extend([
            {'Patient': i,
             'Group': group,
             'Week': time,
             'Response': val} for i, val in enumerate(response)
        ])

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(12, 6))
clinical_raincloud(df, 'Response', 'Group', 'Week', ax)
ax.set_title('Clinical Trial Results: Treatment vs Placebo')
ax.set_xlabel('Response Score')
ax.set_ylabel('Week')
plt.show()
```

Slide 8: Raincloud Plots for Paired Data Analysis

Paired data analysis requires special consideration in visualization to show both individual changes and group-level patterns. This implementation creates split raincloud plots that effectively display paired observations.

```python
def paired_raincloud_plot(pre_data, post_data, ax, color='blue', pair_lines=True):
    """
    Create raincloud plots for paired before-after data.
    
    Parameters:
    -----------
    pre_data : array-like
        Pre-intervention measurements
    post_data : array-like
        Post-intervention measurements
    ax : matplotlib axis
        Axis to plot on
    color : str
        Base color for the plot
    pair_lines : bool
        Whether to draw lines connecting paired observations
    """
    import matplotlib.patches as patches
    
    # Create mirrored raincloud plot
    def half_violin(data, side, position, color):
        v = ax.violinplot(data, positions=[position], vert=False)
        for b in v['bodies']:
            # Get the center of the plot
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # Modify the paths to create half violin
            if side == 'right':
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], m, np.inf)
            else:
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], -np.inf, m)
            b.set_color(color)
            b.set_alpha(0.3)
    
    # Plot pre-data on left side
    half_violin(pre_data, 'left', 0, color)
    ax.boxplot(pre_data, positions=[0], vert=False, widths=0.2,
               patch_artist=True, showfliers=False)
    
    # Plot post-data on right side
    half_violin(post_data, 'right', 1, color)
    ax.boxplot(post_data, positions=[1], vert=False, widths=0.2,
               patch_artist=True, showfliers=False)
    
    # Add individual points
    for i, (pre, post) in enumerate(zip(pre_data, post_data)):
        y_jitter = np.random.normal(0, 0.02)
        if pair_lines:
            ax.plot([pre, post], [0+y_jitter, 1+y_jitter],
                   color=color, alpha=0.2)
        ax.scatter(pre, 0+y_jitter, color=color, alpha=0.5, s=20)
        ax.scatter(post, 1+y_jitter, color=color, alpha=0.5, s=20)
    
    # Add effect size and p-value
    effect_size = (np.mean(post_data) - np.mean(pre_data)) / \
                 np.std(pre_data)
    t_stat, p_val = stats.ttest_rel(pre_data, post_data)
    
    ax.text(ax.get_xlim()[1], 0.5,
            f'Effect Size: {effect_size:.2f}\np-value: {p_val:.4f}',
            ha='left', va='center')

# Example usage
np.random.seed(42)
n_subjects = 50
pre_treatment = np.random.normal(100, 15, n_subjects)
post_treatment = pre_treatment + 25 + np.random.normal(0, 10, n_subjects)

fig, ax = plt.subplots(figsize=(10, 6))
paired_raincloud_plot(pre_treatment, post_treatment, ax)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Pre', 'Post'])
ax.set_title('Paired Raincloud Plot: Treatment Effect')
plt.show()
```

Slide 9: Integration with Statistical Models

This implementation combines raincloud plots with statistical model outputs, showing both raw data distribution and model-predicted effects with confidence intervals.

```python
def model_based_raincloud(data, x_var, y_var, ax, model_type='linear'):
    """
    Create raincloud plots with statistical model overlay.
    
    Parameters:
    -----------
    data : pandas DataFrame
        Input data
    x_var : str
        Categorical predictor variable
    y_var : str
        Continuous outcome variable
    ax : matplotlib axis
        Axis to plot on
    model_type : str
        Type of statistical model to fit
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    
    # Create basic raincloud plot
    positions = np.arange(len(data[x_var].unique()))
    colors = plt.cm.Set2(np.linspace(0, 1, len(positions)))
    
    for idx, group in enumerate(data[x_var].unique()):
        subset = data[data[x_var] == group][y_var]
        
        # KDE
        sns.kdeplot(data=subset, ax=ax, color=colors[idx],
                   fill=True, alpha=0.3)
        
        # Box plot
        bp = ax.boxplot(subset, positions=[positions[idx]],
                       widths=0.1, patch_artist=True)
        plt.setp(bp['boxes'], facecolor='white')
        
        # Strip plot
        y_jitter = np.random.normal(positions[idx], 0.04, size=len(subset))
        ax.scatter(subset, y_jitter, alpha=0.4, color=colors[idx], s=3)
    
    # Fit statistical model
    if model_type == 'linear':
        X = pd.get_dummies(data[x_var])
        X = add_constant(X)
        model = OLS(data[y_var], X).fit()
        
        # Plot model predictions and CIs
        for idx, group in enumerate(data[x_var].unique()):
            pred = model.params[idx]
            ci = model.conf_int()[idx]
            
            ax.hlines(positions[idx], ci[0], ci[1],
                     colors='red', linewidth=2, alpha=0.7)
            ax.plot(pred, positions[idx], 'ro', markersize=8)
    
    # Add model statistics
    r2 = model.rsquared
    f_stat = model.fvalue
    p_val = model.f_pvalue
    
    stats_text = (f'R² = {r2:.3f}\n'
                 f'F = {f_stat:.2f}\n'
                 f'p = {p_val:.4f}')
    ax.text(ax.get_xlim()[1], ax.get_ylim()[1],
            stats_text, ha='right', va='top')

# Example usage
np.random.seed(42)
n_samples = 100
groups = ['Control', 'Low', 'High']
data = []

for group in groups:
    if group == 'Control':
        effect = 0
    elif group == 'Low':
        effect = 2
    else:
        effect = 4
        
    values = np.random.normal(10 + effect, 1.5, n_samples)
    data.extend([{'Group': group, 'Value': v} for v in values])

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(12, 6))
model_based_raincloud(df, 'Group', 'Value', ax)
ax.set_title('Model-based Raincloud Plot')
plt.show()
```

Slide 10: Advanced Customization for Scientific Publications

This implementation provides publication-ready raincloud plots with customizable aesthetics, statistical annotations, and export capabilities suitable for scientific journals.

```python
def publication_raincloud(data_dict, ax, figsize=(10, 6), style='science',
                         palette='colorblind', sig_test=True):
    """
    Create publication-quality raincloud plots.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of data arrays
    ax : matplotlib axis
        Axis to plot on
    style : str
        Visual style ('science', 'nature', 'cell')
    palette : str
        Color palette name
    sig_test : bool
        Whether to add significance tests
    """
    import seaborn as sns
    
    # Set style parameters based on journal
    if style == 'science':
        plt.style.use(['seaborn-white', 'seaborn-paper'])
        font = {'family': 'Arial',
                'weight': 'normal',
                'size': 8}
    elif style == 'nature':
        plt.style.use(['seaborn-white', 'seaborn-paper'])
        font = {'family': 'Helvetica',
                'weight': 'normal',
                'size': 7}
    
    plt.rc('font', **font)
    plt.rc('axes', linewidth=0.5)
    
    # Get colors from specified palette
    colors = sns.color_palette(palette, n_colors=len(data_dict))
    
    for idx, (group, data) in enumerate(data_dict.items()):
        position = idx
        
        # KDE with careful bandwidth selection
        kde = stats.gaussian_kde(data, bw_method='silverman')
        x_kde = np.linspace(min(data), max(data), 200)
        y_kde = kde(x_kde)
        
        # Plot KDE
        ax.fill_between(x_kde, position, position + y_kde/y_kde.max()/2,
                       alpha=0.3, color=colors[idx])
        
        # Box plot with minimal style
        bp = ax.boxplot(data, positions=[position], widths=0.05,
                       showfliers=False, patch_artist=True)
        plt.setp(bp['boxes'], facecolor='white', alpha=1.0)
        plt.setp(bp['medians'], color='k', linewidth=0.5)
        
        # Add individual points with minimal jitter
        y_jitter = np.random.normal(position, 0.02, size=len(data))
        ax.scatter(data, y_jitter, alpha=0.5, s=2, color=colors[idx])
        
        # Add summary statistics
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
        
        # Add error bars
        ax.hlines(position + 0.15, ci[0], ci[1], colors='k',
                 linewidth=0.5, capsize=2)
        ax.plot(mean, position + 0.15, 'ko', markersize=2)
    
    if sig_test and len(data_dict) > 1:
        add_significance_bars(data_dict, ax, test='mann_whitney')
    
    # Customize appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(range(len(data_dict)))
    ax.set_yticklabels(list(data_dict.keys()))
    
    return ax

def add_significance_bars(data_dict, ax, test='mann_whitney'):
    """Add significance bars between groups"""
    groups = list(data_dict.keys())
    y_max = ax.get_ylim()[1]
    
    for i in range(len(groups)-1):
        for j in range(i+1, len(groups)):
            if test == 'mann_whitney':
                stat, p_val = stats.mannwhitneyu(
                    data_dict[groups[i]],
                    data_dict[groups[j]]
                )
            else:
                stat, p_val = stats.ttest_ind(
                    data_dict[groups[i]],
                    data_dict[groups[j]]
                )
            
            # Determine significance level
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            
            # Draw significance bars
            y = y_max + 0.1 * (j - i)
            x1 = np.mean(data_dict[groups[i]])
            x2 = np.mean(data_dict[groups[j]])
            ax.plot([x1, x2], [y, y], 'k-', linewidth=0.5)
            ax.text((x1 + x2)/2, y + 0.02, sig, ha='center', va='bottom')

# Example usage
np.random.seed(42)
example_data = {
    'Control': np.random.normal(10, 1, 100),
    'Treatment A': np.random.normal(11, 1.2, 100),
    'Treatment B': np.random.normal(12.5, 0.8, 100)
}

fig, ax = plt.subplots(figsize=(8, 6))
publication_raincloud(example_data, ax, style='science')
ax.set_title('Publication-Ready Raincloud Plot')
plt.tight_layout()
plt.show()
```

Slide 11: Real-world Example: Gene Expression Analysis

This implementation demonstrates how to use raincloud plots for visualizing differential gene expression data, including multiple conditions and time points.

```python
def gene_expression_raincloud(expression_data, genes, conditions, ax,
                            log_transform=True):
    """
    Create raincloud plots for gene expression data.
    
    Parameters:
    -----------
    expression_data : pandas DataFrame
        Gene expression data
    genes : list
        List of genes to plot
    conditions : list
        List of experimental conditions
    ax : matplotlib axis
        Axis to plot on
    log_transform : bool
        Whether to log-transform expression values
    """
    import seaborn as sns
    
    # Set up colors
    condition_colors = sns.color_palette("husl", n_colors=len(conditions))
    
    for g_idx, gene in enumerate(genes):
        gene_data = expression_data[expression_data['Gene'] == gene]
        
        for c_idx, condition in enumerate(conditions):
            position = g_idx + c_idx * 0.3
            
            # Get expression values for this gene and condition
            values = gene_data[gene_data['Condition'] == condition]['Expression']
            if log_transform:
                values = np.log2(values + 1)
            
            # Create raincloud components
            # KDE
            sns.kdeplot(data=values, ax=ax, color=condition_colors[c_idx],
                       fill=True, alpha=0.3)
            
            # Box plot
            bp = ax.boxplot(values, positions=[position], widths=0.1,
                          patch_artist=True, showfliers=False)
            plt.setp(bp['boxes'], facecolor='white')
            
            # Strip plot
            y = np.random.normal(position, 0.02, size=len(values))
            ax.scatter(values, y, alpha=0.4, s=3,
                      color=condition_colors[c_idx])
            
            # Add statistics
            if len(conditions) > 1 and c_idx > 0:
                control_values = gene_data[
                    gene_data['Condition'] == conditions[0]
                ]['Expression']
                if log_transform:
                    control_values = np.log2(control_values + 1)
                
                # Calculate fold change and p-value
                fold_change = np.mean(values) - np.mean(control_values)
                _, p_val = stats.ttest_ind(values, control_values)
                
                # Add annotation
                ax.text(ax.get_xlim()[1], position,
                       f'FC: {fold_change:.2f}\np: {p_val:.2e}',
                       ha='left', va='center', fontsize=8)

# Generate example gene expression data
np.random.seed(42)
n_samples = 50
genes = ['Gene A', 'Gene B', 'Gene C']
conditions = ['Control', 'Treatment', 'Treatment+Drug']

data = []
for gene in genes:
    base_exp = np.random.uniform(5, 10)
    for condition in conditions:
        if condition == 'Control':
            effect = 0
        elif condition == 'Treatment':
            effect = 1
        else:
            effect = -0.5
            
        expression = np.random.normal(
            base_exp + effect,
            0.5,
            n_samples
        )
        
        data.extend([{
            'Gene': gene,
            'Condition': condition,
            'Expression': max(0, e)  # Expression can't be negative
        } for e in expression])

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(12, 8))
gene_expression_raincloud(df, genes, conditions, ax)
ax.set_title('Gene Expression Analysis')
plt.show()
```

Slide 12: Animated Raincloud Plots for Time Series Analysis

This implementation creates animated raincloud plots to visualize how distributions evolve over time, particularly useful for longitudinal studies and temporal data analysis.

```python
import matplotlib.animation as animation

def animated_raincloud(time_series_data, times, fig, ax):
    """
    Create animated raincloud plot for temporal data.
    
    Parameters:
    -----------
    time_series_data : dict
        Dictionary with timepoints as keys and data arrays as values
    times : list
        List of timepoints
    fig : matplotlib figure
        Figure object
    ax : matplotlib axis
        Axis object
    """
    def update(frame):
        ax.clear()
        time = times[frame]
        data = time_series_data[time]
        
        # KDE
        kde = stats.gaussian_kde(data)
        x_kde = np.linspace(min(data), max(data), 200)
        y_kde = kde(x_kde)
        
        # Plot components
        ax.fill_between(x_kde, 0, y_kde, alpha=0.3, color='blue')
        
        # Box plot
        bp = ax.boxplot(data, vert=False, positions=[0.5],
                       widths=0.2, patch_artist=True)
        plt.setp(bp['boxes'], facecolor='white')
        
        # Strip plot
        y_jitter = np.random.normal(0.5, 0.02, size=len(data))
        ax.scatter(data, y_jitter, alpha=0.4, color='blue', s=3)
        
        # Add statistics
        mean = np.mean(data)
        std = np.std(data)
        ax.text(ax.get_xlim()[1], 0.8,
                f'Time: {time}\nMean: {mean:.2f}\nStd: {std:.2f}',
                ha='right')
        
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([])
        ax.set_title(f'Distribution Evolution - Time {time}')
    
    anim = animation.FuncAnimation(fig, update,
                                 frames=len(times),
                                 interval=500,
                                 repeat=True)
    return anim

# Example usage
np.random.seed(42)
times = list(range(10))
n_samples = 100

# Generate evolving distribution
time_series_data = {}
for t in times:
    # Distribution parameters change with time
    mean = 5 + np.sin(t/2)  # Oscillating mean
    std = 1 + 0.2 * t       # Increasing variance
    time_series_data[t] = np.random.normal(mean, std, n_samples)

fig, ax = plt.subplots(figsize=(10, 6))
anim = animated_raincloud(time_series_data, times, fig, ax)
plt.close()  # Prevent display of static plot

# To save animation (uncomment to use)
# anim.save('raincloud_animation.gif', writer='pillow')
```

Slide 13: Interactive Raincloud Plots

This implementation creates interactive raincloud plots with hover tooltips and click events, enabling detailed exploration of data points and distributions.

```python
def interactive_raincloud(data_dict, title="Interactive Raincloud Plot"):
    """
    Create an interactive raincloud plot using Plotly.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing groups and their data
    title : str
        Plot title
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(rows=len(data_dict), cols=1,
                       shared_xaxes=True)
    
    colors = px.colors.qualitative.Set3
    
    for idx, (group, data) in enumerate(data_dict.items()):
        # Calculate KDE
        kde = stats.gaussian_kde(data)
        x_kde = np.linspace(min(data), max(data), 200)
        y_kde = kde(x_kde)
        
        # Add KDE trace
        fig.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde + idx,
                fill='tonexty',
                name=f'{group} KDE',
                hoverinfo='skip',
                showlegend=False,
                line=dict(color=colors[idx]),
                opacity=0.3
            ),
            row=idx+1, col=1
        )
        
        # Add box plot
        fig.add_trace(
            go.Box(
                x=data,
                name=group,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(color=colors[idx]),
                line=dict(color='black'),
                hovertemplate=
                "Value: %{x}<br>" +
                "Group: " + group + "<br>" +
                "<extra></extra>"
            ),
            row=idx+1, col=1
        )
        
        # Add statistics
        stats_text = (
            f"Mean: {np.mean(data):.2f}<br>"
            f"Median: {np.median(data):.2f}<br>"
            f"Std: {np.std(data):.2f}"
        )
        
        fig.add_annotation(
            x=max(data),
            y=idx,
            text=stats_text,
            showarrow=False,
            xanchor='left',
            yanchor='bottom'
        )
    
    fig.update_layout(
        title=title,
        showlegend=True,
        height=200*len(data_dict),
        width=800,
        hovermode='closest'
    )
    
    return fig

# Example usage
np.random.seed(42)
interactive_data = {
    'Control': np.random.normal(0, 1, 100),
    'Treatment A': np.random.normal(0.5, 1.2, 100),
    'Treatment B': np.random.normal(1.5, 0.8, 100)
}

fig = interactive_raincloud(interactive_data)
# fig.show()  # Uncomment to display in notebook
```

Slide 14: Additional Resources

*   "RainCloud Plots: A Multi-platform Tool for Robust Data Visualization" [https://arxiv.org/abs/1908.03620](https://arxiv.org/abs/1908.03620)
*   "Beyond Bar and Line Graphs: Time for a New Data Presentation Paradigm" [https://arxiv.org/abs/1901.08939](https://arxiv.org/abs/1901.08939)
*   "Statistical Inference through Data Science: A Modern Dive into R and the Tidyverse" [https://arxiv.org/abs/1912.07796](https://arxiv.org/abs/1912.07796)
*   "Ten Simple Rules for Better Figures" [https://arxiv.org/abs/1404.5676](https://arxiv.org/abs/1404.5676)
*   "Visualization of Statistical Distributions in R" [https://arxiv.org/abs/2002.01404](https://arxiv.org/abs/2002.01404)

